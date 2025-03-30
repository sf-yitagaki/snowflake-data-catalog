import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
import snowflake.snowpark.functions as F
import snowflake.snowpark.types as T
import pandas as pd
import json
import logging
import re # JSON抽出のためにインポート
from datetime import datetime, timedelta
import graphviz # データリネージ可視化用

# --- 構成 ---
# ロガー設定
logging.basicConfig(level=logging.INFO) # INFOレベル以上をログに出力 (DEBUGにするとより詳細)
logger = logging.getLogger(__name__)

# LLMモデルとEmbeddingモデル ワークシート(要件に合わせて変更)
DEFAULT_LLM_MODEL = 'claude-3-5-sonnet' # または 'mixtral-8x7b' など
DEFAULT_EMBEDDING_MODEL = 'voyage-multilingual-2'

EMBEDDING_DIMENSION = 1024 # 次元数
EMBED_FUNCTION_NAME = f"SNOWFLAKE.CORTEX.EMBED_TEXT_{EMBEDDING_DIMENSION}" # 使用する関数名を動的に設定

# --- Snowflake接続 ---
try:
    session = get_active_session()
    CURRENT_DATABASE = session.get_current_database().strip('"')
    CURRENT_SCHEMA = session.get_current_schema().strip('"')
    logger.info(f"Snowflakeセッション取得成功。現在のDB: {CURRENT_DATABASE}, スキーマ: {CURRENT_SCHEMA}")
except Exception as e:
    logger.error(f"Snowflakeセッションの取得に失敗しました: {e}")
    st.error("Snowflakeセッションの取得に失敗しました。アプリが正しく動作しない可能性があります。")
    st.stop() # セッションがないと続行できない

# --- ページ設定 ---
st.set_page_config(page_title="データカタログ", layout='wide')
st.title("データカタログアプリ")

# --- 定数 ---
METADATA_TABLE_NAME = "DATA_CATALOG_METADATA" # メタデータテーブル名
SELECT_OPTION = "<Select>" # データベース選択のデフォルト

# --- メタデータテーブル管理 ---
@st.cache_resource
def create_metadata_table():
    """
    データカタログ用のメタデータテーブルが存在しない場合に作成します。
    要件に合わせて列を定義します。
    """
    try:
        # VARCHARの最大長を考慮
        # 要件: DATABASE_NAME, SCHEMA_NAME, TABLE_NAME, ANALYSIS_IDEAS, EMBEDDING, TABLE_COMMENT, LAST_REFRESHED (8つ)
        # 追加: LIKES (いいね機能用)
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE_NAME} (
            database_name VARCHAR(255),
            schema_name VARCHAR(255),
            table_name VARCHAR(255),
            table_comment VARCHAR(16777216), -- AI生成のテーブル概要 (シンプルな説明)
            analysis_ideas VARCHAR(16777216), -- AI生成の分析アイデア (ユースケース説明)
            embedding VECTOR(FLOAT, {EMBEDDING_DIMENSION}), -- ベクトル (snowflake-arctic-embed-l 1024次元)
            likes INT DEFAULT 0, -- いいね数
            last_refreshed TIMESTAMP_LTZ, -- メタデータ更新日時
            PRIMARY KEY (database_name, schema_name, table_name)
        );
        """
        session.sql(ddl).collect()
        logger.info(f"{METADATA_TABLE_NAME} テーブルの存在を確認または作成しました。")
        return True
    except SnowparkSQLException as e:
        # 権限不足などのエラーメッセージをより具体的に表示
        if "does not exist or not authorized" in str(e):
             st.error(f"エラー: メタデータテーブル '{METADATA_TABLE_NAME}' が存在しないか、アクセス権限がありません。")
             logger.error(f"メタデータテーブルアクセスエラー: {e}")
        # VECTOR型が有効でない場合のエラーハンドリングを追加
        elif "Unsupported feature 'VECTOR'" in str(e):
            st.error("エラー: ご利用のSnowflake環境でVECTOR型がサポートされていません。このアプリは利用できません。")
            logger.error(f"VECTOR型非サポートエラー: {e}")
        else:
             st.error(f"メタデータテーブルの準備中にエラーが発生しました: {e}")
             logger.error(f"{METADATA_TABLE_NAME} テーブルの作成/確認中にエラーが発生しました: {e}")
        return False
    except Exception as e:
        logger.error(f"予期せぬエラー (create_metadata_table): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        return False

# --- データ取得関数 (account_usageベースは<コード>から流用) ---

@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_databases():
    """
    アカウント内のデータベース一覧を取得します (account_usage または SHOW DATABASES を使用)。
    先頭に '<Select>' を追加します。
    DELETED_ON または DELETED カラムを試します。
    """
    databases_df = None
    error_occurred = False
    error_message = ""

    # account_usage.databases ビューが存在するか確認 (権限チェック含む)
    try:
        session.sql("SELECT 1 FROM snowflake.account_usage.databases LIMIT 1").collect()
        account_usage_accessible = True
    except SnowparkSQLException as check_err:
        if "does not exist or not authorized" in str(check_err):
            logger.warning(f"account_usage.databases へのアクセス権限エラー: {check_err}")
            account_usage_accessible = False
        else:
            # その他のSQLエラーはaccount_usage利用時に処理させる
             account_usage_accessible = True # 利用を試みる
             logger.warning(f"account_usage.databases の存在チェックで予期せぬエラー: {check_err}")


    if account_usage_accessible:
        try:
            # まず DELETED_ON を試す
            logger.info("account_usage.databases から DELETED_ON を使ってデータベース一覧を取得試行...")
            databases_df = session.sql("""
                SELECT database_name
                FROM snowflake.account_usage.databases
                WHERE deleted_on IS NULL
                ORDER BY database_name
            """).to_pandas()
            logger.info("DELETED_ON を使った取得に成功。")
        except SnowparkSQLException as e1:
            if "invalid identifier 'DELETED_ON'" in str(e1):
                logger.warning("DELETED_ON が見つかりません。DELETED を試します。")
                try:
                    # DELETED_ON がなければ DELETED を試す
                    logger.info("account_usage.databases から DELETED を使ってデータベース一覧を取得試行...")
                    databases_df = session.sql("""
                        SELECT database_name
                        FROM snowflake.account_usage.databases
                        WHERE deleted IS NULL
                        ORDER BY database_name
                    """).to_pandas()
                    logger.info("DELETED を使った取得に成功。")
                except SnowparkSQLException as e2:
                    logger.error(f"DELETED を使ったデータベース一覧取得も失敗しました: {e2}")
                    error_occurred = True
                    error_message = f"account_usage.databases からのデータベース一覧取得に失敗しました (DELETED_ON/DELETED): {e2}"
                except Exception as e_generic_deleted:
                     logger.error(f"DELETED を使ったデータベース一覧取得中に予期せぬエラー: {e_generic_deleted}")
                     error_occurred = True
                     error_message = f"account_usage.databases からのデータベース一覧取得中に予期せぬエラー (DELETED): {e_generic_deleted}"
            else:
                # DELETED_ON 以外のSQLエラー
                logger.error(f"DELETED_ON を使ったデータベース一覧取得中にSQLエラー: {e1}")
                error_occurred = True
                error_message = f"account_usage.databases からのデータベース一覧取得中にSQLエラー: {e1}"
        except Exception as e_generic_deleted_on:
             logger.error(f"DELETED_ON を使ったデータベース一覧取得中に予期せぬエラー: {e_generic_deleted_on}")
             error_occurred = True
             error_message = f"account_usage.databases からのデータベース一覧取得中に予期せぬエラー (DELETED_ON): {e_generic_deleted_on}"

    # account_usage が使えない、またはエラーが発生した場合、SHOW DATABASES を試す
    if databases_df is None or error_occurred:
        if not account_usage_accessible:
             st.warning("`snowflake.account_usage.databases` へのアクセス権限がないため、`SHOW DATABASES` を使用します。")
        elif error_occurred:
             st.warning(f"account_usageからの取得に失敗したため (`{error_message}`), `SHOW DATABASES` を使用します。")

        try:
            logger.info("SHOW DATABASES を使ってデータベース一覧を取得試行...")
            databases_show_result = session.sql("SHOW DATABASES").collect()
            if databases_show_result:
                # 'name' カラムが存在するか確認
                if databases_show_result[0].__contains__("name"):
                     db_names = [row['name'] for row in databases_show_result]
                     databases_df = pd.DataFrame({'DATABASE_NAME': sorted(db_names)})
                     logger.info("SHOW DATABASES を使った取得に成功。")
                     error_occurred = False # SHOW DATABASESで成功したのでエラーフラグ解除
                else:
                    logger.error("SHOW DATABASES の結果に 'name' カラムが含まれていません。")
                    error_occurred = True
                    error_message = "SHOW DATABASES の結果形式が予期されたものではありません。"
            else:
                logger.warning("SHOW DATABASES の結果が空でした。")
                databases_df = pd.DataFrame({'DATABASE_NAME': []}) # 空のDataFrame
                error_occurred = False # 結果が空なのはエラーではない

        except Exception as show_err:
            logger.error(f"SHOW DATABASES の実行に失敗しました: {show_err}")
            # account_usageもSHOW DATABASESも失敗した場合
            st.error(f"データベース一覧の取得に失敗しました。account_usageアクセス試行時のエラー: {error_message}, SHOW DATABASES試行時のエラー: {show_err}")
            return [SELECT_OPTION] # 取得不可

    # 取得成功時の処理
    if databases_df is not None and not error_occurred:
        # データベース名のリストを作成し、先頭に SELECT_OPTION を追加
        db_list = [SELECT_OPTION] + databases_df['DATABASE_NAME'].tolist()
        logger.info(f"{len(db_list) - 1} 件のデータベースを取得しました。")
        return db_list
    else:
        # フォールバックも失敗した場合
        st.error(f"データベース一覧の取得に最終的に失敗しました。エラー: {error_message}")
        return [SELECT_OPTION] # エラー時もデフォルト選択肢を返す

def is_safe_identifier(identifier: str) -> bool:
    """
    識別子が英数字とアンダースコアのみで構成されているか簡易チェックします。
    Snowflakeの識別子ルールはより複雑ですが、基本的な安全性を確認します。
    """
    if not identifier:
        return False
    # 正規表現でより厳密にチェックすることも可能
    # 例: return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_$]*$', identifier))
    # ここでは簡易的に、危険な文字が含まれていないかチェック
    forbidden_chars = ['"', "'", ';', '--', '/*', '*/']
    if any(char in identifier for char in forbidden_chars):
        return False
    # Snowflakeでは引用符で囲めば多くの文字が使えるが、ここでは基本的なケースを想定
    # 必要であれば、より厳密な検証ロジックを追加
    return True

@st.cache_data(ttl=600) # 10分キャッシュ
def get_schemas_for_database(database_name: str):
    """指定されたデータベース内のスキーマ一覧を取得します。"""
    if not database_name or database_name == SELECT_OPTION:
        return []

    # データベース名の安全性をチェック
    if not is_safe_identifier(database_name):
        st.error(f"不正なデータベース名が指定されました: {database_name}")
        logger.error(f"get_schemas_for_database: 不正なデータベース名 {database_name}")
        return []

    try:
        # f-string を使用してデータベース名を埋め込む
        query = f"""
        SELECT schema_name
        FROM {database_name}.INFORMATION_SCHEMA.SCHEMATA
        WHERE schema_name NOT IN ('INFORMATION_SCHEMA', 'PUBLIC')
        ORDER BY schema_name;
        """
        # パラメータは不要なので削除
        schemas_df = session.sql(query).to_pandas()

        schema_list = schemas_df['SCHEMA_NAME'].tolist()
        logger.info(f"データベース '{database_name}' から {len(schema_list)} 件のスキーマを取得しました。")
        return schema_list
    except SnowparkSQLException as e:
        st.warning(f"データベース '{database_name}' のスキーマ取得中にエラーが発生しました: {e}")
        logger.warning(f"get_schemas_for_database エラー ({database_name}): {e}")
        return []
    except Exception as e:
        st.error(f"スキーマ一覧の取得中に予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"get_schemas_for_database 予期せぬエラー ({database_name}): {e}")
        return []

@st.cache_data(ttl=600) # 10分キャッシュ
def get_tables_for_database_schema(database_name: str, selected_schemas: tuple = None): # selected_schemasをタプルに変更 for cache
    """指定されたデータベースとスキーマ（任意）のテーブルとビュー一覧を取得します。"""
    if not database_name or database_name == SELECT_OPTION:
        return pd.DataFrame()

    # データベース名の安全性をチェック
    if not is_safe_identifier(database_name):
        st.error(f"不正なデータベース名が指定されました: {database_name}")
        logger.error(f"get_tables_for_database_schema: 不正なデータベース名 {database_name}")
        return pd.DataFrame()
    # スキーマ名の安全性もチェック (タプルの各要素をチェック)
    if selected_schemas:
        if not all(is_safe_identifier(s) for s in selected_schemas):
             st.error(f"不正なスキーマ名が含まれています: {selected_schemas}")
             logger.error(f"get_tables_for_database_schema: 不正なスキーマ名 {selected_schemas}")
             return pd.DataFrame()

    try:
        # f-string を使用してデータベース名を埋め込む
        query = f"""
        SELECT
            TABLE_CATALOG AS DATABASE_NAME,
            TABLE_SCHEMA AS SCHEMA_NAME,
            TABLE_NAME,
            TABLE_TYPE,
            COMMENT AS SOURCE_TABLE_COMMENT,
            ROW_COUNT,
            BYTES,
            CREATED,
            LAST_ALTERED
        FROM {database_name}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA'
        """
        # paramsからdatabase_nameを削除
        params = []

        if selected_schemas:
            schema_placeholders = ', '.join(['?'] * len(selected_schemas))
            query += f" AND TABLE_SCHEMA IN ({schema_placeholders})"
            params.extend(selected_schemas) # スキーマ名をパラメータとして追加

        query += " ORDER BY TABLE_SCHEMA, TABLE_NAME;"

        tables_df = session.sql(query, params=params).to_pandas() # paramsにはスキーマ名のみ渡す
        schema_str = f"スキーマ {selected_schemas}" if selected_schemas else "全スキーマ"
        logger.info(f"データベース '{database_name}' ({schema_str}) から {len(tables_df)} 件のテーブル/ビューを取得しました。")
        return tables_df

    except SnowparkSQLException as e:
        st.warning(f"データベース '{database_name}' のテーブル/ビュー取得中にエラーが発生しました: {e}")
        logger.warning(f"get_tables_for_database_schema エラー ({database_name}, {selected_schemas}): {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"テーブル/ビュー一覧の取得中に予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"get_tables_for_database_schema 予期せぬエラー ({database_name}, {selected_schemas}): {e}")
        return pd.DataFrame()


# --- メタデータ取得・更新関数 ---
@st.cache_data(ttl=300) # 5分キャッシュ
def get_metadata(database_name, schema_name, table_name):
    """
    指定されたテーブルのメタデータを取得します。
    """
    if not METADATA_TABLE_NAME: return {}
    try:
        # テーブル存在チェック (より堅牢に)
        try:
            session.table(f"{CURRENT_DATABASE}.{CURRENT_SCHEMA}.{METADATA_TABLE_NAME}")
        except SnowparkSQLException:
            logger.warning(f"{METADATA_TABLE_NAME} テーブルが見つかりません。メタデータを取得できません。")
            # テーブル作成を試みるか、エラーメッセージを出す
            if not create_metadata_table():
                 st.error(f"メタデータテーブル {METADATA_TABLE_NAME} の準備に失敗しました。")
                 return {}
            # テーブル作成後、再度取得を試みる (ただしキャッシュのため初回は空が返る可能性)
            return {}

        query = f"""
        SELECT
            TABLE_COMMENT, -- AI生成コメント
            ANALYSIS_IDEAS,
            EMBEDDING,
            LIKES,
            LAST_REFRESHED
        FROM {METADATA_TABLE_NAME}
        WHERE DATABASE_NAME = ? AND SCHEMA_NAME = ? AND TABLE_NAME = ?;
        """
        result = session.sql(query, params=[database_name, schema_name, table_name]).to_pandas()
        if not result.empty:
            meta = result.iloc[0].to_dict()
            # EMBEDDING は JSON 文字列として保存されている場合があるため、リストに変換
            if 'EMBEDDING' in meta and isinstance(meta['EMBEDDING'], str):
                try:
                    # SnowflakeのVECTOR型は直接Pythonリストとして返されるはずだが、念のため
                    meta['EMBEDDING'] = json.loads(meta['EMBEDDING'])
                except json.JSONDecodeError:
                    logger.warning(f"EMBEDDINGカラムのJSONデコード失敗: {database_name}.{schema_name}.{table_name}")
                    meta['EMBEDDING'] = None
            return meta
        else:
            return {}
    except SnowparkSQLException as e:
        logger.warning(f"メタデータ取得中にエラー ({database_name}.{schema_name}.{table_name}): {e}")
        return {}
    except Exception as e:
        logger.error(f"予期せぬエラー (get_metadata): {e}")
        return {}


def update_metadata(database_name, schema_name, table_name, data_dict):
    """
    指定されたテーブルのメタデータを更新または挿入します。
    MERGE文でのVECTOR型設定方法を修正 (PARSE_JSON と CAST を使用)。
    """
    if not METADATA_TABLE_NAME: return False
    try:
        update_clauses = []
        insert_cols = ["database_name", "schema_name", "table_name", "last_refreshed"]
        insert_vals = ["source.database_name", "source.schema_name", "source.table_name", "CURRENT_TIMESTAMP()"]
        source_cols = ["? AS database_name", "? AS schema_name", "? AS table_name"]
        params_base = [database_name, schema_name, table_name] # 基本パラメータ
        params_dynamic = [] # 動的に追加されるパラメータ
        embedding_param_json = None # Embedding 用の JSON 文字列パラメータ

        # --- 各カラムの処理 ---
        if 'TABLE_COMMENT' in data_dict:
            update_clauses.append("table_comment = source.table_comment")
            insert_cols.append("table_comment")
            insert_vals.append("source.table_comment")
            source_cols.append("? AS table_comment")
            params_dynamic.append(data_dict['TABLE_COMMENT'])
        if 'ANALYSIS_IDEAS' in data_dict:
            update_clauses.append("analysis_ideas = source.analysis_ideas")
            insert_cols.append("analysis_ideas")
            insert_vals.append("source.analysis_ideas")
            source_cols.append("? AS analysis_ideas")
            params_dynamic.append(data_dict['ANALYSIS_IDEAS'])

        if 'EMBEDDING' in data_dict:
            embedding_list = data_dict['EMBEDDING']
            # VECTOR型にキャストするSQL式 (パラメータプレースホルダを使用)
            # PARSE_JSONでJSON文字列をVARIANTに変換し、それをVECTORにキャスト
            embedding_sql_expr = f"(PARSE_JSON(?))::VECTOR(FLOAT, {EMBEDDING_DIMENSION})"

            if embedding_list is not None and isinstance(embedding_list, list):
                 # 有効なベクトルリストがある場合
                 update_clauses.append(f"embedding = {embedding_sql_expr}")
                 insert_cols.append("embedding")
                 insert_vals.append(embedding_sql_expr)
                 # パラメータとして渡すJSON文字列を準備
                 embedding_param_json = json.dumps(embedding_list)
            else:
                 # Embedding を NULL に設定する場合、または無効な値の場合
                 update_clauses.append("embedding = NULL")
                 # INSERT 時は embedding 列を含めない (デフォルト NULL になる)
                 logger.warning(f"Embedding for {table_name} is None or not a list, setting to NULL.")


        # --- LIKES 列の処理 ---
        if 'LIKES_INCREMENT' in data_dict and data_dict['LIKES_INCREMENT']:
            update_clauses.append("likes = target.likes + 1")
        elif 'LIKES' in data_dict:
             update_clauses.append("likes = source.likes")
             insert_cols.append("likes")
             insert_vals.append("source.likes")
             source_cols.append("? AS likes")
             params_dynamic.append(data_dict['LIKES'])

        # --- MERGE 文の構築 ---
        # 更新句がなく、Embedding更新もない場合（Likesインクリメントのみなど）でも処理実行
        # if not update_clauses and 'EMBEDDING' not in data_dict: ... このチェックは不要かも

        # 常にlast_refreshedは更新する
        update_clauses.append("last_refreshed = CURRENT_TIMESTAMP()")

        merge_sql = f"""
        MERGE INTO {METADATA_TABLE_NAME} AS target
        USING (SELECT {', '.join(source_cols)}) AS source
        ON target.database_name = source.database_name
           AND target.schema_name = source.schema_name
           AND target.table_name = source.table_name
        WHEN MATCHED THEN
            UPDATE SET {', '.join(update_clauses)}
        WHEN NOT MATCHED THEN
            INSERT ({', '.join(insert_cols)})
            VALUES ({', '.join(insert_vals)});
        """

        # --- パラメータリストの最終化 ---
        # 基本パラメータ + 動的パラメータ + Embeddingパラメータ (あれば)
        final_params = params_base + params_dynamic
        param_count_in_sql = merge_sql.count('?') # SQL文中のプレースホルダ数をカウント

        # Embedding パラメータが必要な回数だけ追加されるように調整
        expected_embedding_params = 0
        if embedding_param_json is not None:
            if f"embedding = {embedding_sql_expr}" in merge_sql:
                expected_embedding_params += 1
            if embedding_sql_expr in insert_vals:
                expected_embedding_params += 1

            for _ in range(expected_embedding_params):
                final_params.append(embedding_param_json)

        # 最終的なパラメータ数とSQL内のプレースホルダ数が一致するか確認 (デバッグ用)
        if len(final_params) != param_count_in_sql:
             logger.error(f"Parameter count mismatch! SQL needs {param_count_in_sql}, but got {len(final_params)}.")
             logger.error(f"SQL: {merge_sql}")
             logger.error(f"Params: {final_params}")
             st.error("内部エラー: メタデータ更新時のパラメータ数が一致しません。ログを確認してください。")
             return False


        # --- SQL 実行 ---
        logger.debug(f"Executing MERGE SQL for {table_name}: {merge_sql}")
        logger.debug(f"With Params ({len(final_params)} items): {final_params}") # パラメータ確認用ログ
        session.sql(merge_sql, params=final_params).collect()

        logger.info(f"メタデータを更新/挿入しました: {database_name}.{schema_name}.{table_name}, updated_keys: {list(data_dict.keys())}")
        st.cache_data.clear()
        return True

    except SnowparkSQLException as e:
        logger.error(f"メタデータ更新中にSQLエラー ({table_name}): {e}", exc_info=True)
        # 失敗したSQLとパラメータをログに出力（パラメータは一部マスクした方が良い場合もある）
        try:
            # final_params が定義されているか確認
            log_params = final_params if 'final_params' in locals() else params_base + params_dynamic
            logger.error(f"Failed SQL: {merge_sql}")
            logger.error(f"Failed Params: {log_params}")
        except NameError:
             logger.error("Failed to log SQL/Params due to NameError.")

        st.error(f"メタデータ更新中にSQLエラー ({table_name})。ログを確認してください。エラー: {e}")
        return False
    except Exception as e:
        logger.error(f"メタデータ更新中に予期せぬエラー ({table_name}): {e}", exc_info=True)
        st.error(f"メタデータ更新中に予期せぬエラー ({table_name}) が発生しました: {e}")
        return False


@st.cache_data(ttl=300) # 5分キャッシュ
def get_all_metadata():
    """
    メタデータテーブルから全てのメタデータを取得します。
    """
    if not METADATA_TABLE_NAME: return {}
    try:
        # テーブル存在チェック
        try:
            session.table(f"{CURRENT_DATABASE}.{CURRENT_SCHEMA}.{METADATA_TABLE_NAME}")
        except SnowparkSQLException:
            logger.warning(f"{METADATA_TABLE_NAME} テーブルが見つかりません。メタデータを取得できません。")
            if not create_metadata_table():
                 st.error(f"メタデータテーブル {METADATA_TABLE_NAME} の準備に失敗しました。")
                 return {}
            return {} # 作成直後は空

        query = f"SELECT * FROM {METADATA_TABLE_NAME};"
        metadata_df = session.sql(query).to_pandas()
        metadata_dict = {}
        for _, row in metadata_df.iterrows():
            key = f"{row['DATABASE_NAME']}.{row['SCHEMA_NAME']}.{row['TABLE_NAME']}"
            meta = row.to_dict()
            # Embeddingの処理 (get_metadataと同様)
            if 'EMBEDDING' in meta and isinstance(meta['EMBEDDING'], str):
                try:
                    meta['EMBEDDING'] = json.loads(meta['EMBEDDING'])
                except json.JSONDecodeError:
                    logger.warning(f"get_all_metadata: EMBEDDINGカラムのJSONデコード失敗: {key}")
                    meta['EMBEDDING'] = None
            metadata_dict[key] = meta
        logger.info(f"{len(metadata_dict)} 件のメタデータを取得しました。")
        return metadata_dict
    except SnowparkSQLException as e:
        logger.error(f"全メタデータ取得中にエラー: {e}")
        st.error(f"メタデータの取得中にエラーが発生しました: {e}")
        return {}
    except Exception as e:
        logger.error(f"予期せぬエラー (get_all_metadata): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        return {}

# --- アクセス数取得関数 (新規追加) ---
@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_monthly_access_count(database_name, schema_name, table_name):
    """
    指定されたテーブルの直近1ヶ月間のアクセス数を取得します。
    ACCOUNT_USAGE.ACCESS_HISTORY を使用します。
    """
    try:
        one_month_ago = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        full_table_name = f"{database_name}.{schema_name}.{table_name}".upper()

        # ACCESS_HISTORYビューが存在するか確認
        try:
            session.sql("SELECT 1 FROM snowflake.account_usage.access_history LIMIT 1").collect()
        except SnowparkSQLException as check_err:
            if "does not exist or not authorized" in str(check_err):
                logger.warning(f"snowflake.account_usage.access_history へのアクセス権限がないため、アクセス数を取得できません。")
                return "N/A" # 権限がない場合は N/A を返す
            else:
                raise # その他のSQLエラー

        # クエリ: 対象テーブルが直接または基本オブジェクトとしてアクセスされた回数をカウント
        # 注意: このクエリは大規模な環境ではコストがかかる可能性があります。
        #       OBJECTS_MODIFIED など他のカラムも考慮する必要があるかもしれません。
        query = f"""
        SELECT COUNT(*) as access_count
        FROM snowflake.account_usage.access_history
        WHERE query_start_time >= '{one_month_ago}'
        AND (
             ARRAY_CONTAINS('{full_table_name}'::variant, direct_objects_accessed)
             OR
             ARRAY_CONTAINS('{full_table_name}'::variant, base_objects_accessed)
        );
        """

        result = session.sql(query).collect()
        if result:
            count = result[0]['ACCESS_COUNT']
            logger.debug(f"アクセス数取得成功 ({full_table_name}): {count}")
            return count
        else:
            logger.warning(f"アクセス数の取得結果が空でした ({full_table_name})")
            return 0
    except SnowparkSQLException as e:
        logger.error(f"アクセス数取得中にSQLエラー ({database_name}.{schema_name}.{table_name}): {e}")
        # エラーによっては 'N/A' を返す方が良いかもしれない
        return "エラー"
    except Exception as e:
        logger.error(f"アクセス数取得中に予期せぬエラー ({database_name}.{schema_name}.{table_name}): {e}")
        return "エラー"


# --- LLM連携関数 ---
@st.cache_data(ttl=3600) # 1時間キャッシュ
def get_table_schema(database_name, schema_name, table_name):
    """指定されたテーブルのカラム情報を取得します。"""
    # 識別子の安全性をチェック
    if not is_safe_identifier(database_name) or \
       not is_safe_identifier(schema_name) or \
       not is_safe_identifier(table_name):
        st.error(f"不正な識別子が含まれています: DB={database_name}, SC={schema_name}, TBL={table_name}")
        logger.error(f"get_table_schema: 不正な識別子 DB={database_name}, SC={schema_name}, TBL={table_name}")
        return pd.DataFrame()

    try:
        # f-string を使用してデータベース名を埋め込む
        # schema_name と table_name はパラメータとして渡す
        full_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT
        FROM {database_name}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION;
        """
        # paramsから database_name を削除
        params = [schema_name, table_name]
        schema_df = session.sql(full_query, params=params).to_pandas()

        if schema_df.empty:
             logger.warning(f"スキーマ情報が見つかりません: {database_name}.{schema_name}.{table_name}")
        else:
             logger.info(f"スキーマ情報を取得: {database_name}.{schema_name}.{table_name}")
        return schema_df
    except SnowparkSQLException as e:
        if "does not exist or not authorized" in str(e):
             logger.warning(f"テーブルが見つからないか権限がありません: {database_name}.{schema_name}.{table_name}")
             # スキーマ情報がない場合もエラーではなく空のDataFrameを返す
        else:
             logger.error(f"スキーマ情報取得エラー ({database_name}.{schema_name}.{table_name}): {e}")
             # UIにエラーを出すと処理が中断される可能性があるため、警告に留めるか検討
             st.warning(f"テーブル '{database_name}.{schema_name}.{table_name}' のスキーマ情報取得中にエラーが発生しました: {e}")
        return pd.DataFrame() # エラー時も空のDataFrameを返す
    except Exception as e:
        logger.error(f"予期せぬエラー (get_table_schema): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        return pd.DataFrame()
        return pd.DataFrame()


def generate_comment_and_ideas(database_name, schema_name, table_name, source_table_comment, model=DEFAULT_LLM_MODEL):
    """
    LLMを使用してテーブルの簡潔なコメントと分析アイデアを生成します。
    session.call を使用して Cortex 関数を呼び出します。
    """
    try:
        schema_df = get_table_schema(database_name, schema_name, table_name)
        if schema_df.empty:
            # スキーマ情報がない場合は警告をログに出力し、AI生成をスキップ
            logger.warning(f"スキーマ情報が取得できなかったため、AI生成をスキップします: {database_name}.{schema_name}.{table_name}")
            # UIに警告を出すと大量に出る可能性があるため、ログのみとする
            # st.warning(f"スキーマ情報が取得できなかったため、AI生成をスキップします: {database_name}.{schema_name}.{table_name}")
            return None, None # Noneを返して呼び出し元で処理

        schema_text = "カラム名 | データ型 | NULL許容 | カラムコメント\n------- | -------- | -------- | --------\n"
        for _, row in schema_df.iterrows():
            col_name = row['COLUMN_NAME'] if pd.notna(row['COLUMN_NAME']) else ""
            data_type = row['DATA_TYPE'] if pd.notna(row['DATA_TYPE']) else ""
            nullable = row['IS_NULLABLE'] if pd.notna(row['IS_NULLABLE']) else ""
            comment_str = str(row['COMMENT']) if pd.notna(row['COMMENT']) and row['COMMENT'] else ""
            schema_text += f"{col_name} | {data_type} | {nullable} | {comment_str}\n"

        source_comment_text = f"\n既存のテーブルコメント: {source_table_comment}" if source_table_comment and pd.notna(source_table_comment) else ""

        prompt = f"""
        あなたはデータカタログ作成を支援するAIです。
        以下のテーブルスキーマ情報と既存コメントに基づいて、このテーブルがどのようなデータを持っているかの「簡潔なテーブルコメント(100字以内)」と、このテーブルデータを使った具体的な「分析アイデア/ユースケース」を3つ提案してください。
        
        テーブル名: {database_name}.{schema_name}.{table_name}
        {source_comment_text}
        
        スキーマ情報:
        ```sql
        {schema_text}
        ```
        
        応答は以下のJSON形式で、「簡潔なテーブルコメント(table_comment)」と「分析アイデア(analysis_ideas)」のみを日本語で返してください。他のテキストは含めないでください。
        
        {{
        "table_comment": "(ここにテーブルデータの簡潔な説明を100字以内で記述)",
        "analysis_ideas": [
        "(ここに分析アイデア/ユースケース1を具体的に記述)",
        "(ここに分析アイデア/ユースケース2を具体的に記述)",
        "(ここに分析アイデア/ユースケース3を具体的に記述)"
        ]
        }}
        """
        sql_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)"
        params = [model, prompt]

        try:
            logger.info(f"Cortex Complete呼び出し開始 ({database_name}.{schema_name}.{table_name}), model={model}, query='{sql_query}'")
            # session.sql(...).collect() を実行
            response = session.sql(sql_query, params=params).collect()
            logger.info(f"Cortex Complete呼び出し完了 ({database_name}.{schema_name}.{table_name})")
        
            # 結果の処理 (collect() はリストを返すので、その要素にアクセス)
            if response and response[0] and response[0][0]:
                result_json_str = response[0][0]
        
                # (JSON抽出、パース処理は変更なし)
                try:
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', result_json_str, re.DOTALL)
                    if json_match:
                        result_json_str = next(g for g in json_match.groups() if g is not None)
                    else:
                        start_index = result_json_str.find('{')
                        end_index = result_json_str.rfind('}')
                        if start_index != -1 and end_index != -1 and start_index < end_index:
                            result_json_str = result_json_str[start_index:end_index+1]
                        else:
                            raise ValueError("LLM応答からJSON部分を抽出できませんでした。")
                except (ValueError, AttributeError) as extract_err:
                    logger.error(f"LLM応答からJSON部分の抽出に失敗 ({database_name}.{schema_name}.{table_name}): {extract_err}. Raw response: {result_json_str[:500]}...")
                    st.error("LLMからの応答形式が不正です。JSON部分を抽出できませんでした。")
                    return "AIコメント生成失敗", ["AI分析アイデア生成失敗"]
        
                logger.info(f"LLM応答 (抽出後JSON) ({database_name}.{schema_name}.{table_name}): {result_json_str[:200]}...")
                try:
                    result_data = json.loads(result_json_str)
                    generated_comment = result_data.get("table_comment")
                    ideas = result_data.get("analysis_ideas")
        
                    if not generated_comment or not isinstance(generated_comment, str) or not generated_comment.strip():
                        logger.warning(f"LLM応答からtable_commentが取得/検証できませんでした。応答: {result_json_str}")
                        generated_comment = "AIコメント生成失敗"
                    if not isinstance(ideas, list) or not ideas:
                        logger.warning(f"LLM応答のanalysis_ideasがリスト形式でないか空です。応答: {result_json_str}")
                        ideas = ["AI分析アイデア生成失敗"]
                    elif not all(isinstance(idea, str) and idea.strip() for idea in ideas):
                        logger.warning(f"LLM応答のanalysis_ideasに不正な要素が含まれます。応答: {result_json_str}")
                        ideas = [idea for idea in ideas if isinstance(idea, str) and idea.strip()]
                        if not ideas: ideas = ["AI分析アイデア生成失敗"]

                    
                    return generated_comment, ideas
                except json.JSONDecodeError:
                    logger.error(f"LLM応答のJSONパースに失敗 ({database_name}.{schema_name}.{table_name}): {result_json_str}")
                    st.error("LLMからの応答の解析に失敗しました。")
                    return "AIコメント生成失敗", ["AI分析アイデア生成失敗"]
                except Exception as parse_err:
                    logger.error(f"LLM応答の処理中にエラー ({database_name}.{schema_name}.{table_name}): {parse_err}")
                    st.error("LLM応答の処理中に予期せぬエラーが発生しました。")
                    return "AIコメント生成失敗", ["AI分析アイデア生成失敗"]
            else:
                logger.error(f"LLMからの応答が空です ({database_name}.{schema_name}.{table_name})")
                st.error("LLMからの応答がありませんでした。")
                return None, None
        except SnowparkSQLException as e:
            logger.error(f"LLM呼び出しエラー ({database_name}.{schema_name}.{table_name}), Query: '{sql_query}', Params: [{model}, <prompt>]", exc_info=True) # クエリとエラー詳細をログに
            # エラーメッセージをUIに表示
            st.error(f"LLM (Complete) 呼び出し中にSQLエラーが発生しました。ログを確認してください。エラー: {e}")
            return None, None
    except Exception as e:
        logger.error(f"予期せぬエラー (generate_comment_and_ideas): {e}", exc_info=True) # 詳細なトレースバックをログに
        st.error(f"予期せぬエラーが発生しました: {e}")
        return None, None



def generate_embedding(text, model=DEFAULT_EMBEDDING_MODEL): # model引数のデフォルト値を更新
    """
    与えられたテキストのベクトル表現を生成します。
    モデル名とテキストを引数として渡すように修正。
    """
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("ベクトル生成のための有効なテキストがありません。")
        return None

    # session.sql を使用し、モデル名とテキストの2つの引数を渡す
    sql_query = f"SELECT {EMBED_FUNCTION_NAME}(?, ?)"
    params = [model, text]

    try:
        # モデル名、関数名、クエリをログに出力
        logger.info(f"Cortex Embed呼び出し開始 (model={model}, func={EMBED_FUNCTION_NAME}, query='{sql_query}')")
        # session.sql(...).collect() を実行
        result_df = session.sql(sql_query, params=params).collect()
        logger.info(f"Cortex Embed呼び出し完了")

        # 結果の処理 (collect() はリストを返す)
        if result_df and result_df[0] and result_df[0][0]:
            embedding_vector = result_df[0][0]
            # (以降のベクトル形式チェック、エラーハンドリングは変更なし)
            if isinstance(embedding_vector, str):
                try:
                    embedding_vector = json.loads(embedding_vector)
                except json.JSONDecodeError as e:
                    logger.error(f"ベクトル結果のJSONデコードエラー: {e}, 元の文字列: {embedding_vector[:100]}...")
                    st.error("ベクトル生成結果の解析に失敗しました。")
                    return None

            if isinstance(embedding_vector, list) and len(embedding_vector) == EMBEDDING_DIMENSION:
                logger.info(f"テキストのベクトル生成に成功しました (次元数: {len(embedding_vector)})")
                return embedding_vector
            else:
                 logger.error(f"生成されたベクトルの形式または次元数が不正です。型: {type(embedding_vector)}, 次元数: {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}, 期待値: {EMBEDDING_DIMENSION}")
                 st.error(f"生成されたベクトルの形式/次元数が不正です (期待値: {EMBEDDING_DIMENSION})。")
                 return None
        else:
            logger.error("ベクトル生成の結果が空です。")
            st.error("ベクトル生成の結果がありませんでした。")
            return None
    except SnowparkSQLException as e:
        logger.error(f"ベクトル生成中にSnowpark SQLエラーが発生しました: Query='{sql_query}', Params=[{model}, <text>]", exc_info=True)
        # モデル名に関するエラーの可能性を追記
        if "invalid identifier" in str(e) or "does not exist or not authorized" in str(e):
             logger.error(f"Cortex関数 {EMBED_FUNCTION_NAME} の呼び出しエラー: {e}")
             st.error(f"ベクトル生成関数({EMBED_FUNCTION_NAME})が見つからないか、権限がありません。")
        elif "Unknown model" in str(e) or "not found" in str(e): # モデルが見つからない場合のエラー例
            logger.error(f"指定されたモデル '{model}' が見つかりません: {e}")
            st.error(f"ベクトル生成モデル '{model}' が見つかりません。サポートされているモデル名か確認してください。")
        else:
            st.error(f"ベクトル生成中にSQLエラーが発生しました。ログを確認してください。エラー: {e}")
        return None
    except Exception as e:
        logger.error(f"ベクトル生成中に予期せぬエラーが発生しました: {e}", exc_info=True)
        st.error(f"ベクトル生成中に予期せぬエラーが発生しました: {e}")
        return None


def generate_and_save_ai_metadata(database_name, schema_name, table_name, source_table_comment):
    """
    LLMでコメントとアイデアを生成し、ベクトルも生成してメタデータテーブルに保存します。
    """
    progress_text = f"{database_name}.{schema_name}.{table_name}: AIコメントと分析アイデアを生成中..."
    spinner_placeholder = st.spinner(progress_text)

    # 1. コメントとアイデアを生成
    generated_comment, ideas = generate_comment_and_ideas(database_name, schema_name, table_name, source_table_comment)

    if generated_comment is None or ideas is None:
        spinner_placeholder.empty()
        st.error(f"'{table_name}' のAIコメントまたは分析アイデアの生成に失敗しました。")
        return False
    if generated_comment == "AIコメント生成失敗":
        spinner_placeholder.empty()
        st.warning(f"'{table_name}' のAIコメント生成に失敗しました。分析アイデアのみ保存を試みます。")
        # アイデアのみ保存に進む

    # 2. ベクトルを生成 (生成されたコメントに基づいて)
    embedding = None
    # if generated_comment and generated_comment != "AIコメント生成失敗":
    if generated_comment:
        # spinner_placeholder.text(f"{database_name}.{schema_name}.{table_name}: ベクトルを生成中...")
        embedding = generate_embedding(generated_comment)

        
        if embedding is None:
            st.warning(f"'{table_name}' のベクトル生成に失敗しました。コメントとアイデアのみ保存します。")
    else:
        st.warning(f"'{table_name}': AIコメントが生成されなかったため、ベクトルは生成されません。")

    # 3. メタデータテーブルに保存
    # spinner_placeholder.text(f"{database_name}.{schema_name}.{table_name}: メタデータを保存中...")
    update_data = {
        "TABLE_COMMENT": generated_comment, # AIが生成したコメント
        "ANALYSIS_IDEAS": json.dumps(ideas, ensure_ascii=False), # JSON文字列として保存
        "EMBEDDING": embedding # 生成されたベクトル (Noneの場合もある)
        # LIKES はここでは更新しない
        # SOURCE_TABLE_COMMENT は情報スキーマから取得するものなので、ここでは保存しない
    }

    # update_metadata を呼び出す前に spinner を終了させる
    # spinner_placeholder.empty()
    if update_metadata(database_name, schema_name, table_name, update_data):
        st.success(f"'{table_name}' のメタデータ (コメント, アイデア, ベクトル) を生成・保存しました。")
        # キャッシュクリア
        st.cache_data.clear()
        return True
    else:
        st.error(f"'{table_name}' のメタデータ保存に失敗しました。")
        return False


# --- データリネージ関連関数 ---
@st.cache_data(ttl=1800) # 30分キャッシュ
def get_dynamic_lineage(target_database, target_schema, target_table, direction='upstream', max_depth=3, time_window_days=90):
    """
    指定されたテーブルを起点として、ACCESS_HISTORYから動的な依存関係（データの流れ）を取得します。
    Args:
        target_database (str): 起点テーブルのデータベース名
        target_schema (str): 起点テーブルのスキーマ名
        target_table (str): 起点テーブル名
        direction (str): 'upstream' (データの入力元を辿る) のみサポート
        max_depth (int): 遡る最大のステップ数
        time_window_days (int): 検索対象とする期間（日数）
    Returns:
        dict: {'nodes': list, 'edges': list} またはエラー時に None
    """
    if direction != 'upstream':
        st.warning("現在、動的リネージは上流方向（upstream）のみサポートしています。")
        return None

    logger.info(f"動的リネージ取得開始: {target_database}.{target_schema}.{target_table}, max_depth={max_depth}, days={time_window_days}")
    nodes = set() # (node_id, domain)
    edges = set() # (source_id, target_id, query_id)
    start_node_id = f"{target_database}.{target_schema}.{target_table}".upper()
    nodes_to_process = [(start_node_id, 0)] # (node_id, current_depth)
    processed_nodes = set() # 無限ループ防止

    # 必要なビューへのアクセス権限チェック
    required_views = ["snowflake.account_usage.access_history", "snowflake.account_usage.query_history"]
    for view_name in required_views:
        try:
            session.sql(f"SELECT 1 FROM {view_name} LIMIT 1").collect()
        except SnowparkSQLException as check_err:
            if "does not exist or not authorized" in str(check_err):
                logger.error(f"{view_name} へのアクセス権限エラー: {check_err}")
                st.error(f"データリネージ取得に必要な `{view_name}` へのアクセス権限がありません。")
                return None
            else:
                logger.error(f"{view_name} のチェック中にエラー: {check_err}")
                st.error(f"データリネージ情報の取得中にエラーが発生しました: {check_err}")
                return None
        except Exception as e:
             logger.error(f"{view_name} チェック中に予期せぬエラー: {e}", exc_info=True)
             st.error(f"データリネージ情報の取得中に予期せぬエラーが発生しました: {e}")
             return None

    # 起点ノードのドメインを取得
    try:
        domain_query = f"SELECT TABLE_TYPE FROM {target_database}.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? LIMIT 1;"
        params = [target_schema, target_table]
        domain_res = session.sql(domain_query, params=params).collect()
        start_node_domain = domain_res[0]['TABLE_TYPE'] if domain_res else 'TABLE' # デフォルトTABLE
        nodes.add((start_node_id, start_node_domain))
    except Exception as e:
        logger.warning(f"起点ノードのドメイン取得失敗 ({start_node_id}): {e}")
        nodes.add((start_node_id, 'TABLE'))

    # 検索期間の計算
    start_time_limit = (datetime.utcnow() - timedelta(days=time_window_days)).strftime('%Y-%m-%d %H:%M:%S')

    while nodes_to_process:
        current_node_id, current_depth = nodes_to_process.pop(0)

        if current_depth >= max_depth or current_node_id in processed_nodes:
            continue

        processed_nodes.add(current_node_id)
        logger.debug(f"深度 {current_depth} で探索中: {current_node_id}")

        try:
            # current_node_id に書き込んだクエリと、そのクエリが読み取ったソースを取得
            # direct_objects_accessed と base_objects_accessed の両方を見る
            # パフォーマンスのため、必要なカラムのみを選択し、サブクエリを使用
            # テーブル名に特殊文字が含まれる可能性を考慮し、完全修飾名を適切に扱う
            # 潜在的なエラーを防ぐため、NULLチェックを追加
            query = f"""
            WITH ModifiedObjects AS (
                -- 対象オブジェクトに書き込んだクエリを特定
                SELECT DISTINCT ah.query_id
                FROM snowflake.account_usage.access_history ah
                CROSS JOIN TABLE(FLATTEN(input => ah.objects_modified)) mod_obj
                WHERE ah.query_start_time >= '{start_time_limit}'
                  AND mod_obj.value:"objectName"::string = '{current_node_id}' -- 完全修飾名で比較
                  AND mod_obj.value:"objectId" IS NOT NULL -- objectId が NULL でないもの
            ), AccessedSources AS (
                -- 上記クエリがアクセスしたオブジェクト（ソース）を特定
                SELECT
                    mo.query_id,
                    acc.value:"objectName"::string as source_object_name,
                    acc.value:"objectDomain"::string as source_object_domain
                FROM snowflake.account_usage.access_history ah
                JOIN ModifiedObjects mo ON ah.query_id = mo.query_id
                CROSS JOIN TABLE(FLATTEN(input => COALESCE(ah.direct_objects_accessed, PARSE_JSON('[]')))) acc -- NULLの場合空配列
                WHERE acc.value:"objectName"::string IS NOT NULL
                  AND acc.value:"objectName"::string != '{current_node_id}'

                UNION -- UNIONで重複排除

                SELECT
                    mo.query_id,
                    base_acc.value:"objectName"::string as source_object_name,
                    base_acc.value:"objectDomain"::string as source_object_domain
                FROM snowflake.account_usage.access_history ah
                JOIN ModifiedObjects mo ON ah.query_id = mo.query_id
                CROSS JOIN TABLE(FLATTEN(input => COALESCE(ah.base_objects_accessed, PARSE_JSON('[]')))) base_acc -- NULLの場合空配列
                WHERE base_acc.value:"objectName"::string IS NOT NULL
                  AND base_acc.value:"objectName"::string != '{current_node_id}'
                  AND base_acc.value:"objectDomain"::string NOT IN ('Stage', 'Query') -- StageやQuery自体は除外
            )
            -- 結果を選択 (重複排除済み)
            SELECT query_id, source_object_name, source_object_domain
            FROM AccessedSources
            WHERE source_object_name IS NOT NULL;
            """

            results_df = session.sql(query).to_pandas()
            logger.debug(f"{current_node_id} への書き込み元クエリ結果: {len(results_df)} 件")

            for _, row in results_df.iterrows():
                # 結果がNoneでないことを確認
                source_id_full_val = row.get('SOURCE_OBJECT_NAME')
                source_domain_val = row.get('SOURCE_OBJECT_DOMAIN')
                query_id = row.get('QUERY_ID')

                if not source_id_full_val or not query_id:
                    logger.warning(f"ソースオブジェクト名またはQuery IDがNULLのためスキップ: {row.to_dict()}")
                    continue

                source_id_full = str(source_id_full_val).upper()
                source_domain = str(source_domain_val).upper() if source_domain_val else 'UNKNOWN'


                # 有効なソースかチェック (完全修飾名か？など簡易チェック)
                if len(source_id_full.split('.')) < 3:
                    logger.warning(f"無効なソースオブジェクト名をスキップ: {source_id_full} (from query {query_id})")
                    continue

                # ノードとエッジを追加
                nodes.add((source_id_full, source_domain))
                edge = (source_id_full, current_node_id, query_id) # エッジにクエリIDを含める

                if edge not in edges:
                    edges.add(edge)
                    logger.debug(f"新しい動的エッジ発見: {source_id_full} -> {current_node_id} (Query: {query_id})")
                    # 次の探索対象に追加
                    if current_depth + 1 < max_depth:
                         # まだ処理されておらず、探索キューにもないノードのみ追加
                         if source_id_full not in processed_nodes and not any(n[0] == source_id_full for n in nodes_to_process):
                            nodes_to_process.append((source_id_full, current_depth + 1))

        except SnowparkSQLException as e:
             logger.error(f"動的リネージクエリ中にSQLエラー ({current_node_id}): {e}", exc_info=True)
             st.warning(f"リネージ情報の一部取得中にSQLエラーが発生しました: {e}")
        except Exception as e:
             logger.error(f"動的リネージ取得中に予期せぬエラー ({current_node_id}): {e}", exc_info=True)
             st.warning(f"リネージ情報の一部取得中に予期せぬエラーが発生しました: {e}")

    # 結果を整形して返す
    result_nodes = [{'id': n[0], 'label': n[0].split('.')[-1], 'domain': n[1]} for n in nodes]
    # エッジ情報にquery_idを追加
    result_edges = [{'source': e[0], 'target': e[1], 'query_id': e[2]} for e in edges]

    logger.info(f"動的リネージ取得完了: ノード数={len(result_nodes)}, エッジ数={len(result_edges)}")
    return {'nodes': result_nodes, 'edges': result_edges}


# Graphviz描画関数もエッジラベル（Query IDなど）に対応させる
def create_lineage_graph(nodes, edges, start_node_id):
    """
    ノードとエッジのリストからGraphvizオブジェクトを生成します。
    エッジにQuery IDのツールチップを追加します。
    """
    # Graphvizの依存関係に関する注意は main ブロックに移動
    dot = graphviz.Digraph(comment='Data Lineage', graph_attr={'rankdir': 'LR'}) # 動的リネージはLR (左から右) が見やすいことが多い

    shape_map = {
        'TABLE': 'box',
        'VIEW': 'ellipse',
        'MATERIALIZED VIEW': 'box3d',
        'STREAM': 'cds',
        'TASK': 'component',
        'PIPE': 'cylinder',
        'FUNCTION': 'septagon',
        'PROCEDURE': 'octagon',
        'STAGE': 'folder',
        'EXTERNAL TABLE': 'tab',
        'UNKNOWN': 'question',
    }

    processed_node_ids = set()
    for node in nodes:
        node_id = node['id']
        if node_id in processed_node_ids: continue # 重複描画防止
        processed_node_ids.add(node_id)

        node_label = node['label']
        node_domain = node.get('domain', 'UNKNOWN').upper()
        shape = shape_map.get(node_domain, 'ellipse')
        node_tooltip = f"Type: {node_domain}\nID: {node_id}"

        if node_id == start_node_id.upper():
            dot.node(node_id, label=node_label, shape=shape, style='filled', fillcolor='lightcoral', tooltip=node_tooltip)
        else:
            dot.node(node_id, label=node_label, shape=shape, style='filled', fillcolor='lightblue', tooltip=node_tooltip)

    processed_edges = set()
    for edge in edges:
        # sourceとtargetが同じエッジは描画しない
        if edge['source'] == edge['target']:
            continue

        edge_tuple = (edge['source'], edge['target'])
        if edge_tuple in processed_edges: continue # 重複描画防止
        processed_edges.add(edge_tuple)

        query_id = edge.get('query_id')
        edge_tooltip = f"Source: {edge['source']}\nTarget: {edge['target']}"
        if query_id:
            edge_tooltip += f"\nQuery ID: {query_id}"
        # エッジラベルは省略（グラフが複雑になるため）、ツールチップで情報提供
        dot.edge(edge['source'], edge['target'], tooltip=edge_tooltip)

    return dot


def display_table_card(table_info, metadata):
    """
    テーブル情報をカード形式で表示。テーブル名サイズ変更。
    table_info に基本情報（DB, Schema, Table名）があれば動作するように修正。
    """
    # --- 必須情報の取得 ---
    db_name = table_info.get('DATABASE_NAME')
    sc_name = table_info.get('SCHEMA_NAME')
    tbl_name = table_info.get('TABLE_NAME')

    if not db_name or not sc_name or not tbl_name:
        logger.error(f"display_table_card に必須情報 (DB, Schema, Table名) が不足しています: {table_info}")
        return

    table_key = f"{db_name}.{sc_name}.{tbl_name}"
    elem_key_base = re.sub(r'\W+', '_', table_key)

    if not isinstance(metadata, dict):
        logger.warning(f"display_table_cardに不正な型のmetadataが渡されました: {type(metadata)} for key {table_key}")
        metadata = {}

    card = st.container(border=True)

    # --- カードヘッダー: テーブル名(サイズ小)といいねボタン ---
    col1, col2 = card.columns([0.85, 0.15])
    with col1:
        col1.markdown(f"**{db_name}.{sc_name}.{tbl_name}**")
    with col2:
        current_likes = metadata.get("LIKES", 0)
        if col2.button(f"👍 {current_likes}", key=f"like_{elem_key_base}", help="いいね！"):
            if update_metadata(db_name, sc_name, tbl_name, {"LIKES_INCREMENT": True}):
                 st.toast(f"「{tbl_name}」にいいねしました！", icon="👍")
                 st.cache_data.clear()
                 st.rerun()
            else:
                 st.error("いいねの更新に失敗しました。")

    # --- 検索類似度 (あれば表示) ---
    search_similarity = table_info.get('search_similarity')
    if search_similarity is not None and pd.notna(search_similarity):
         card.caption(f"検索キーワードとの関連度: {search_similarity:.2%}")

    # --- LLMによるコメントとアイデア ---
    llm_comment = metadata.get("TABLE_COMMENT", None)
    analysis_ideas_str = metadata.get("ANALYSIS_IDEAS", "[]")
    analysis_ideas = []
    if analysis_ideas_str and isinstance(analysis_ideas_str, str):
        try:
            ideas_parsed = json.loads(analysis_ideas_str)
            if isinstance(ideas_parsed, list):
                analysis_ideas = ideas_parsed
            else:
                 analysis_ideas = ["(分析アイデアの形式が不正)"]
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"ANALYSIS_IDEASのJSONデコード失敗: {table_key}")
            analysis_ideas = ["(分析アイデアの形式が不正)"]

    # --- 詳細表示 (Expander) ---
    with card.expander("詳細を表示"):
        st.markdown("**LLMによるテーブル概要:**")
        if llm_comment and "生成失敗" not in str(llm_comment):
            st.caption(llm_comment)
        elif "生成失敗" in str(llm_comment):
             st.caption(f"AIによる概要生成に失敗しました。({llm_comment})") # エラー詳細も表示
        else:
            st.caption("未生成")
            # AI生成ボタンは information_schema 由来のコメントがなくても押せるようにする
            source_comment_orig = table_info.get('SOURCE_TABLE_COMMENT') # 元のコメント取得試行
            if st.button("LLMで概要と分析アイデアを生成", key=f"gen_ai_{elem_key_base}"):
                 # 元コメントがなくても None を渡して実行
                 generate_and_save_ai_metadata(db_name, sc_name, tbl_name, source_comment_orig)
                 st.rerun()

        st.markdown("**LLMによる分析アイデア/ユースケース:**")
        if analysis_ideas and not any("生成失敗" in str(idea) for idea in analysis_ideas):
            for idea in analysis_ideas:
                st.caption(f"- {idea}")
        elif any("生成失敗" in str(idea) for idea in analysis_ideas):
             st.caption(f"LLMによる分析アイデア生成に失敗しました。({analysis_ideas})")
        elif llm_comment and "生成失敗" not in str(llm_comment):
             st.caption("未生成")
        else:
            st.caption("未生成 (概要と同時に生成されます)")

        st.divider()

        # --- 元のテーブルコメント (INFORMATION_SCHEMA) ---
        source_comment_display = table_info.get('SOURCE_TABLE_COMMENT')
        if source_comment_display and pd.notna(source_comment_display):
            st.markdown("**元のテーブルコメント (Information Schema):**")
            st.caption(source_comment_display)
            st.divider()


        # --- アクセス数とテーブル情報 ---
        col_meta1, col_meta2 = st.columns(2)

        with col_meta1:
             st.markdown("**直近1ヶ月のアクセス数:**")
             monthly_access = get_monthly_access_count(db_name, sc_name, tbl_name)
             st.metric(label="アクセス回数 (クエリ単位)", value=monthly_access)

        with col_meta2:
            st.markdown("**テーブル情報:**")
            info_dict = {
                "タイプ": table_info.get('TABLE_TYPE', 'N/A'), # infoにあれば表示
                "行数": table_info.get('ROW_COUNT', 'N/A'),   # infoにあれば表示
                "サイズ(Bytes)": table_info.get('BYTES', 'N/A'), # infoにあれば表示
                "作成日時": table_info.get('CREATED'),      # infoにあれば表示
                "最終更新日時": table_info.get('LAST_ALTERED'), # infoにあれば表示
                "メタデータ最終更新": metadata.get('LAST_REFRESHED') # metadataから取得
            }
            display_info = {}
            for k, v in info_dict.items():
                 # None や N/A を除外して表示を整形
                 if v is not None and v != 'N/A' and pd.notna(v):
                    if isinstance(v, (int, float)):
                        if k == "行数":
                            display_info[k] = f"{v:,.0f}" if v >= 0 else "N/A"
                        elif k == "サイズ(Bytes)":
                            if v >= 1024**3: display_info[k] = f"{v / 1024**3:.2f} GB"
                            elif v >= 1024**2: display_info[k] = f"{v / 1024**2:.2f} MB"
                            elif v >= 1024: display_info[k] = f"{v / 1024:.2f} KB"
                            elif v >= 0: display_info[k] = f"{v} Bytes"
                            else: display_info[k] = "N/A"
                        else:
                            display_info[k] = v
                    elif isinstance(v, (datetime, pd.Timestamp)):
                        try:
                            if v.tzinfo: display_info[k] = v.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
                            else: display_info[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception: display_info[k] = str(v)
                    else:
                         display_info[k] = str(v)

            # 表示項目がある場合のみ表示
            if display_info:
                for key, val in display_info.items():
                    st.text(f"{key}: {val}")
            else:
                st.caption("追加情報なし")
                
         # --- データリネージ (動的依存関係) ---
        st.divider()
        st.markdown("**データリネージ（データの流れ - 上流）**")
        # 動的リネージに関する説明と設定項目を追加
        lineage_days = st.number_input("遡る日数", min_value=1, max_value=365, value=90, step=1, key=f"lineage_days_{elem_key_base}", help="この日数前までのデータ操作履歴（ACCESS_HISTORY）を検索します。長くすると時間がかかります。")
        lineage_depth = st.number_input("遡る深さ", min_value=1, max_value=5, value=3, step=1, key=f"lineage_depth_{elem_key_base}", help="何ステップ前までデータの流れを遡るか指定します。深くすると時間がかかります。")

        st.caption(f"`ACCESS_HISTORY`ビューに基づき、過去{lineage_days}日間のデータの流れ（どのテーブル/ビューからデータが来たか）を表示します。最大{3}時間程度のデータ遅延が発生する場合があります。検索範囲が広いと時間がかかることがあります。")

        lineage_visible_key = f"lineage_visible_{elem_key_base}"
        lineage_button_placeholder = st.empty()

        if not st.session_state.get(lineage_visible_key, False):
            if lineage_button_placeholder.button("リネージを表示/更新", key=f"lineage_btn_show_{elem_key_base}"):
                st.session_state[lineage_visible_key] = True
                st.rerun()

        if st.session_state.get(lineage_visible_key, False):
            if lineage_button_placeholder.button("リネージを非表示", key=f"lineage_btn_hide_{elem_key_base}"):
                st.session_state[lineage_visible_key] = False
                st.rerun()

            lineage_placeholder = st.empty()
            with lineage_placeholder.container():
                st.info(f"過去{lineage_days}日間のリネージ情報を最大{lineage_depth}ステップ遡って取得・描画中...")
                try:
                    # 動的リネージ取得関数を呼び出し (日数と深さを渡す)
                    lineage_data = get_dynamic_lineage(db_name, sc_name, tbl_name, direction='upstream', max_depth=lineage_depth, time_window_days=lineage_days)

                    if lineage_data is None:
                         # get_dynamic_lineage内で権限エラー等表示済み
                         st.warning("リネージ情報の取得に失敗しました。権限や設定を確認してください。")
                    elif lineage_data['nodes'] and len(lineage_data['nodes']) > 1:
                        start_node_full_id = f"{db_name}.{sc_name}.{tbl_name}".upper()
                        graph = create_lineage_graph(lineage_data['nodes'], lineage_data['edges'], start_node_full_id)
                        st.graphviz_chart(graph)
                    elif lineage_data['nodes']: # 起点ノードのみ
                         st.info(f"過去{lineage_days}日間の履歴では、このオブジェクトの上流データは見つかりませんでした。")
                    else: # データ取得成功だが空
                        st.warning("リネージ情報の取得に失敗しました（データが空）。")

                except ImportError:
                     logger.error("Graphvizライブラリが見つかりません。")
                     st.error("リネージ表示に必要なGraphvizライブラリ/実行ファイルが環境にない可能性があります。")
                except graphviz.backend.execute.ExecutableNotFound:
                     logger.error("Graphviz実行ファイルが見つかりません。")
                     st.error("Graphviz実行ファイルが見つかりません。環境設定を確認してください。")
                except Exception as e:
                     logger.error(f"リネージ表示エラー ({table_key}): {e}", exc_info=True)
                     st.error(f"リネージ表示中に予期せぬエラー: {e}")


def main_page():
    st.header("データカタログ")

    if 'metadata_table_created' not in st.session_state:
        if create_metadata_table():
            st.session_state.metadata_table_created = True
        else:
            st.error("メタデータテーブルの準備に失敗したため、処理を中断します。")
            return

    # --- サイドバー ---
    st.sidebar.header("1. 検索 & フィルター")
    search_term = st.sidebar.text_input("キーワード検索 (全テーブル対象)", key="search_input")
    search_vector = st.sidebar.toggle("ベクトル検索を有効にする(0.2程度を推奨)", value=True, help=f"検索語とAI生成コメントのベクトル類似度({DEFAULT_EMBEDDING_MODEL}, {EMBEDDING_DIMENSION}次元)で検索します。", key="search_vector_toggle")
    similarity_threshold = 0.2
    if search_vector:
        # 閾値スライダーを追加（任意）
        similarity_threshold = st.sidebar.slider("類似度の閾値", 0.0, 1.0, 0.2, 0.05, key="similarity_slider")
    search_button = st.sidebar.button("検索実行", key="search_button")
    st.sidebar.divider()
    st.sidebar.header("2. 特定のテーブルを検索")
    db_options = get_databases()
    selected_db = st.sidebar.selectbox("データベースで絞り込み", options=db_options, index=0, key="db_select")
    selected_schemas = []
    if selected_db != SELECT_OPTION:
        schema_options = get_schemas_for_database(selected_db)
        if schema_options:
             selected_schemas = st.sidebar.multiselect("スキーマで絞り込み", options=schema_options, default=[], key="schema_select")

    # --- メイン表示エリア ---
    status_placeholder = st.empty()
    results_container = st.container()

    # --- 検索実行時の処理 ---
    if search_button and search_term:
        status_placeholder.info("全テーブルを対象に検索を実行中...")
        search_lower = search_term.lower()
        final_results_df = pd.DataFrame() # 最終的な表示結果DF
        vector_search_executed = False # ベクトル検索を試みたか
        vector_search_succeeded = False # ベクトル検索SQLが成功したか

        try:
            # --- クエリとパラメータの準備 ---
            select_columns_base = """
                database_name, schema_name, table_name, table_comment,
                analysis_ideas, embedding, likes, last_refreshed
            """

            # --- ベクトル検索が有効な場合の処理 ---
            if search_vector:
                vector_search_executed = True # ベクトル検索を試みたフラグ
                status_placeholder.text("検索語のベクトルを計算中...")
                escaped_search_term = search_term.replace("'", "''")

                # CTEを使ったベクトル検索SQL (類似度を SIMILARITY として取得)
                # テーブルエイリアスを削除し、シンプルに
                vector_search_sql = f"""
                WITH search_vector AS (
                    SELECT {EMBED_FUNCTION_NAME}(?, ?) as query_embedding
                )
                SELECT
                    {select_columns_base},
                    VECTOR_COSINE_SIMILARITY(embedding, sv.query_embedding) as SIMILARITY
                FROM
                    {METADATA_TABLE_NAME}, search_vector sv
                WHERE
                    embedding IS NOT NULL
                ORDER BY
                    similarity DESC NULLS LAST -- SQLレベルでソートしておく
                """
                vector_params = [DEFAULT_EMBEDDING_MODEL, escaped_search_term]

                logger.info(f"Executing vector search query (using CTE): {vector_search_sql}")
                logger.debug(f"Vector search params: {vector_params}")

                # ベクトル検索SQLを実行
                vector_results_df = session.sql(vector_search_sql, params=vector_params).to_pandas()
                vector_search_succeeded = True # SQL自体は成功
                logger.info(f"Vector search query returned {len(vector_results_df)} results.")
                logger.debug(f"Vector search results columns: {vector_results_df.columns.tolist()}")

                # --- Python側でのフィルタリング (類似度閾値のみ) ---
                status_placeholder.text("検索結果をフィルタリング中...")
                filtered_rows = []
                if not vector_results_df.empty and 'SIMILARITY' in vector_results_df.columns:
                    for index, row in vector_results_df.iterrows():
                        similarity = row.get('SIMILARITY', 0.0)
                        similarity = similarity if pd.notna(similarity) else 0.0

                        if similarity >= similarity_threshold:
                            row_dict = row.to_dict()
                            # 表示用の類似度を `search_similarity` として追加
                            row_dict['search_similarity'] = similarity
                            filtered_rows.append(row_dict)

                    final_results_df = pd.DataFrame(filtered_rows)
                    # SQLでソート済みだが、念のため再ソートしても良い
                    # if not final_results_df.empty:
                    #     final_results_df = final_results_df.sort_values(by='search_similarity', ascending=False)
                else:
                     logger.warning("Vector search results missing 'SIMILARITY' column or empty.")
                     final_results_df = pd.DataFrame() # 結果なし

            # --- ベクトル検索が無効な場合の処理 (キーワード検索のみ) ---
            else:
                keyword_search_sql = f"""
                SELECT
                    {select_columns_base}
                    , NULL as SIMILARITY -- 類似度列をNULLで追加しておく（列構造を合わせるため）
                FROM {METADATA_TABLE_NAME}
                WHERE
                    (
                        LOWER(database_name) LIKE ? OR LOWER(schema_name) LIKE ? OR
                        LOWER(table_name) LIKE ? OR LOWER(table_comment) LIKE ? OR
                        LOWER(analysis_ideas) LIKE ?
                    )
                ORDER BY database_name, schema_name, table_name
                """
                keyword_param = f"%{search_lower}%"
                keyword_params = [keyword_param] * 5 # 5つのLIKE条件に対応

                logger.info(f"Executing keyword search query: {keyword_search_sql}")
                logger.debug(f"Keyword search params: {keyword_params}")
                final_results_df = session.sql(keyword_search_sql, params=keyword_params).to_pandas()
                logger.info(f"Keyword search query returned {len(final_results_df)} results.")
                # キーワード検索結果に search_similarity 列を追加して None を設定
                final_results_df['search_similarity'] = None


        except SnowparkSQLException as e:
            status_placeholder.error(f"検索クエリの実行中にSQLエラーが発生しました: {e}")
            logger.error(f"Search query execution failed: {e}", exc_info=True)
            failed_sql = vector_search_sql if vector_search_executed else keyword_search_sql
            failed_params = vector_params if vector_search_executed else keyword_params
            logger.error(f"Failed SQL: {failed_sql}")
            logger.error(f"Failed Params: {failed_params}")
            final_results_df = pd.DataFrame() # エラー時は空
        except Exception as e:
            status_placeholder.error(f"検索処理中に予期せぬエラーが発生しました: {e}")
            logger.error(f"Unexpected error during search: {e}", exc_info=True)
            final_results_df = pd.DataFrame() # エラー時は空

        # --- 検索中メッセージを消す ---
        status_placeholder.empty()

        # --- 検索結果の表示 (final_results_df を使用) ---
        with results_container:
            st.subheader(f"検索結果: '{search_term}'")
            search_info = f"{len(final_results_df)} 件のテーブルが見つかりました。"
            if vector_search_executed and vector_search_succeeded:
                 search_info += f" (類似度 >= {similarity_threshold} でフィルタ)"
            elif vector_search_executed and not vector_search_succeeded:
                 search_info += " (ベクトル検索失敗、キーワード検索のみ実行)"
            st.info(search_info)

            if not final_results_df.empty:
                all_metadata_display = get_all_metadata() # メタデータは別途取得
                cols = st.columns(3)
                col_idx = 0
                # final_results_df をループ
                for index, row_data in final_results_df.iterrows():
                    db_name_search = row_data.get('DATABASE_NAME')
                    sc_name_search = row_data.get('SCHEMA_NAME')
                    tbl_name_search = row_data.get('TABLE_NAME')

                    if not db_name_search or not sc_name_search or not tbl_name_search:
                        logger.warning(f"Search result row (index {index}) is missing required keys (DB/SC/TBL): {row_data}")
                        continue

                    table_info_search = {
                        'DATABASE_NAME': db_name_search,
                        'SCHEMA_NAME': sc_name_search,
                        'TABLE_NAME': tbl_name_search,
                        # 'search_similarity' 列から値を取得（キーワード検索時はNone）
                        'search_similarity': row_data.get('search_similarity')
                    }
                    table_key_search = f"{db_name_search}.{sc_name_search}.{tbl_name_search}"
                    metadata_search = all_metadata_display.get(table_key_search, {})

                    with cols[col_idx % 3]:
                        display_table_card(table_info_search, metadata_search)
                    col_idx += 1
            elif search_term:
                 st.warning("指定されたキーワード/類似度に一致するテーブルは見つかりませんでした。")

    # --- 検索ボタンが押されていない、または検索語がない場合の処理 ---
    else:
        if selected_db != SELECT_OPTION:
            # (変更なし)
            status_placeholder.info(f"データベース '{selected_db}' のテーブルを表示中...")
            schemas_to_fetch_browse = tuple(selected_schemas) if selected_schemas else None
            tables_df_browse = get_tables_for_database_schema(selected_db, schemas_to_fetch_browse)
            status_placeholder.empty()
            with results_container:
                st.subheader(f"テーブル一覧: {selected_db}{f' (スキーマ: {selected_schemas})' if selected_schemas else ''}")
                st.info(f"{len(tables_df_browse)} 件のテーブル/ビューが見つかりました。")
                if not tables_df_browse.empty:
                    all_metadata_browse = get_all_metadata()
                    cols_browse = st.columns(3)
                    col_idx_browse = 0
                    for index, table_row_browse in tables_df_browse.iterrows():
                        try:
                            table_info_browse = {k: v for k, v in table_row_browse.items() if pd.notna(v)}
                            if not table_info_browse.get('DATABASE_NAME') or \
                               not table_info_browse.get('SCHEMA_NAME') or \
                               not table_info_browse.get('TABLE_NAME'):
                                logger.warning(f"Browse result row (index {index}) is missing required keys.")
                                continue
                            table_key_browse = f"{table_info_browse['DATABASE_NAME']}.{table_info_browse['SCHEMA_NAME']}.{table_info_browse['TABLE_NAME']}"
                            metadata_browse = all_metadata_browse.get(table_key_browse, {})
                            with cols_browse[col_idx_browse % 3]:
                                display_table_card(table_info_browse, metadata_browse)
                            col_idx_browse += 1
                        except Exception as e:
                            logger.error(f"Error processing browse result row (index {index}): {e}", exc_info=True)
                else:
                    st.warning("指定された条件に一致するテーブル/ビューはありません。")
        else:
            status_placeholder.info("サイドバーで検索を実行するか、データベースを選択してテーブルを表示してください。")

    
# --- 管理ページ ---
def admin_page():
    st.header("管理機能")

    if 'metadata_table_created' not in st.session_state:
        if create_metadata_table():
            st.session_state.metadata_table_created = True
        else:
            st.error("メタデータテーブルの準備に失敗したため、管理機能を利用できません。")
            return

    st.subheader("LLMによるメタデータ生成")
    st.markdown(f"""
    - 選択したデータベース・スキーマ配下のテーブルについて以下のメタデータをLLMで生成し、メタデータテーブル (`{METADATA_TABLE_NAME}`) に保存します。
        - LLMによるテーブルコメント (`TABLE_COMMENT`)
        - 分析アイデア (`ANALYSIS_IDEAS`)
        - ベクトル (`EMBEDDING`) 
    
  
    **注意:**
    *   既にメタデータが存在する場合、新しい情報で上書きされます。
    *   テーブル数が多い場合、処理に時間がかかり、クレジットを消費します。
    *   `SNOWFLAKE.CORTEX` 関数へのアクセス権限が必要です。
    *   使用モデル:
        *   コメント/アイデア生成: `{DEFAULT_LLM_MODEL}`
        *   ベクトル生成: `{DEFAULT_EMBEDDING_MODEL}` (次元数: {EMBEDDING_DIMENSION})
        *   必要に応じて使用するモデルを変更してください。
    """)

    col1, col2 = st.columns(2)
    with col1:
        db_options_admin = get_databases()
        db_options_admin = [db for db in db_options_admin if db != SELECT_OPTION]
        selected_db_admin = st.selectbox(
            "対象データベースを選択",
            options=db_options_admin,
            index=0 if db_options_admin else -1,
            key="admin_db_select"
        )

    with col2:
        schema_options_admin = []
        selected_schemas_admin = []
        if selected_db_admin:
            schema_options_admin = get_schemas_for_database(selected_db_admin)
            if schema_options_admin:
                 selected_schemas_admin = st.multiselect(
                     "対象スキーマを選択 (未選択時は全スキーマ)",
                     options=schema_options_admin,
                     default=[],
                     key="admin_schema_select"
                 )
            else:
                st.caption(f"'{selected_db_admin}' 内のスキーマを取得できませんでした。")
        else:
             st.caption("データベースを選択してください。")

    if selected_db_admin:
        st.markdown("---")
        st.markdown("**生成対象テーブルのプレビュー**")

        try:
            schemas_to_fetch_admin = tuple(selected_schemas_admin) if selected_schemas_admin else None
            # プレビュー表示のためのテーブル取得 (キャッシュを使う)
            target_tables_df = get_tables_for_database_schema(selected_db_admin, schemas_to_fetch_admin)

            if not target_tables_df.empty:
                # メタデータの存在状況を付加 (キャッシュを使う)
                all_meta_admin = get_all_metadata()
                target_tables_df['metadata_key'] = target_tables_df.apply(
                    lambda row: f"{row['DATABASE_NAME']}.{row['SCHEMA_NAME']}.{row['TABLE_NAME']}",
                    axis=1
                )
                target_tables_df['metadata_exists'] = target_tables_df['metadata_key'].isin(all_meta_admin)
                target_tables_df['llm_comment_exists'] = target_tables_df['metadata_key'].apply(
                    lambda key: all_meta_admin.get(key, {}).get('TABLE_COMMENT') not in [None, "AIコメント生成失敗"]
                )
                target_tables_df['embedding_exists'] = target_tables_df['metadata_key'].apply(
                    lambda key: all_meta_admin.get(key, {}).get('EMBEDDING') is not None
                )

                display_cols = ['DATABASE_NAME', 'SCHEMA_NAME', 'TABLE_NAME', 'TABLE_TYPE', 'llm_comment_exists', 'embedding_exists']
                st.dataframe(target_tables_df[display_cols], use_container_width=True, height=300)

                total_target_count = len(target_tables_df)
                overwrite_count = target_tables_df['metadata_exists'].sum()
                overwrite_notice = f" (うち {overwrite_count} 件は既存メタデータを上書き)" if overwrite_count > 0 else ""

                st.info(f"{total_target_count} 件のテーブルが対象です{overwrite_notice}。")

                # ボタンにユニークなキーを設定
                if st.button(f"{total_target_count} 件のテーブルのメタデータを生成/更新する", type="primary", key="admin_generate_button"):

                    # ループ全体の進捗表示用のウィジェット変数名を変更
                    admin_loop_progress_bar = st.progress(0)
                    admin_loop_status_placeholder = st.empty() # st.empty()でプレースホルダ確保
                    success_count = 0
                    fail_count = 0

                    # ボタンが押された後に再度テーブル一覧を取得（キャッシュされているはずだが念のため）
                    # このdfをループで使う
                    process_tables_df = get_tables_for_database_schema(selected_db_admin, schemas_to_fetch_admin)
                    total_process_count = len(process_tables_df) # 処理件数を再計算

                    for i, (_, table_row) in enumerate(process_tables_df.iterrows()):
                        db = table_row['DATABASE_NAME']
                        sc = table_row['SCHEMA_NAME']
                        tbl = table_row['TABLE_NAME']
                        src_comment = table_row.get('SOURCE_TABLE_COMMENT')


                        admin_loop_status_placeholder.text(f"全体進捗 ({i+1}/{total_process_count}): {db}.{sc}.{tbl} を処理中...")

                        try:
                            # generate_and_save_ai_metadata は内部で st.status を使用し、
                            # 個別のテーブル処理の詳細ステータスを表示する
                            if generate_and_save_ai_metadata(db, sc, tbl, src_comment):
                                success_count += 1
                            else:
                                # generate_and_save_ai_metadata内でエラー表示されるため、ここではカウントのみ
                                fail_count += 1
                        except Exception as e:
                             # 予期せぬエラーが発生した場合
                             fail_count += 1
                             logger.error(f"メタデータ生成ループ内で予期せぬエラー ({db}.{sc}.{tbl}): {e}", exc_info=True)
                             # プレースホルダーにエラー情報を一時的に表示してもよい
                             admin_loop_status_placeholder.error(f"テーブル {tbl} の処理中に予期せぬエラー発生。ログを確認してください。", icon="🔥")
                             # エラーが発生しても次のテーブルの処理に進む

                        # ループ全体のプログレスバーを更新
                        admin_loop_progress_bar.progress((i + 1) / total_process_count)

                    # ループ完了後、ループ用のプログレスバーとメッセージエリアをクリア
                    admin_loop_progress_bar.empty()
                    admin_loop_status_placeholder.empty()

                    # 最終結果メッセージ
                    if success_count > 0:
                        st.success(f"{success_count} 件のテーブルのメタデータ生成/更新が完了しました。")
                    if fail_count > 0:
                        st.warning(f"{fail_count} 件のテーブルの処理に失敗またはスキップされました。詳細はアプリログを確認してください。")
                    # 処理対象が0件だった場合のメッセージは不要かも (ボタン押下前に件数表示されるため)

                    # キャッシュクリアと再描画
                    st.cache_data.clear()
                    st.rerun()

            else:
                st.warning("指定されたデータベース/スキーマにテーブルが見つかりませんでした。")

        except Exception as e:
            st.error(f"対象テーブルの取得またはプレビュー表示中にエラーが発生しました: {e}")
            logger.error(f"管理ページでのテーブル取得/プレビューエラー: {e}", exc_info=True)

    else:
        st.info("メタデータを生成するデータベースを選択してください。")


def main():
    page = st.sidebar.radio(
        "ページ選択",
        ["データカタログ", "管理"],
        key="page_selection"
    )

    if page == "データカタログ":
        main_page()
    elif page == "管理":
        admin_page()


if __name__ == "__main__":
    main()