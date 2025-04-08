# 必要なライブラリをインポート
import streamlit as st  # Streamlit アプリケーションフレームワーク
from snowflake.snowpark.context import get_active_session  # Snowflake Snowpark セッション取得
from snowflake.snowpark.exceptions import SnowparkSQLException  # Snowpark SQL 例外処理
import snowflake.snowpark.functions as F  # Snowpark 関数 (未使用だが、将来的な利用のため残す場合あり)
import snowflake.snowpark.types as T  # Snowpark 型定義 (未使用だが、将来的な利用のため残す場合あり)
import pandas as pd  # データ操作・分析ライブラリ
import json  # JSON データ操作
import logging  # ログ出力
import re  # 正規表現操作 (主にLLM応答からのJSON抽出用)
from datetime import datetime, timedelta  # 日時操作
import graphviz # データリネージ可視化用 (Graphvizライブラリ)

# --- 構成 ---
# ロガー設定: アプリケーションの動作状況を記録するための設定
logging.basicConfig(level=logging.INFO) # INFOレベル以上のログメッセージを出力 (DEBUGにするとより詳細な情報)
logger = logging.getLogger(__name__) # このモジュール用のロガーを取得

# LLMモデルとEmbeddingモデル ワークシート(要件に合わせて変更)
DEFAULT_LLM_MODEL = 'claude-3-5-sonnet' # デフォルトで使用するLLMモデル名
DEFAULT_EMBEDDING_MODEL = 'multilingual-e5-large' # 使用するテキスト埋め込みモデル名
# 備考: 使用するモデルはSnowflake環境で利用可能なものを指定してください。

EMBEDDING_DIMENSION = 1024 # Embedding ベクトルの次元数 (使用するモデルに合わせる)
# 使用する Snowflake Cortex Embed Text 関数の名前を動的に設定
EMBED_FUNCTION_NAME = f"SNOWFLAKE.CORTEX.EMBED_TEXT_{EMBEDDING_DIMENSION}" # モデルの次元数に合わせて関数名が変わる想定

# --- Snowflake接続 ---
try:
    # Streamlit in Snowflake 環境などからアクティブな Snowpark セッションを取得
    session = get_active_session()
    # 現在接続しているデータベース名を取得し、不要な引用符を除去
    CURRENT_DATABASE = session.get_current_database().strip('"')
    # 現在接続しているスキーマ名を取得し、不要な引用符を除去
    CURRENT_SCHEMA = session.get_current_schema().strip('"')
    # 接続成功のログを出力
    logger.info(f"Snowflakeセッション取得成功。現在のDB: {CURRENT_DATABASE}, スキーマ: {CURRENT_SCHEMA}")
except Exception as e:
    # セッション取得に失敗した場合のエラーハンドリング
    logger.error(f"Snowflakeセッションの取得に失敗しました: {e}")
    # Streamlit UI上にエラーメッセージを表示
    st.error("Snowflakeセッションの取得に失敗しました。アプリが正しく動作しない可能性があります。")
    # セッションがないとアプリは動作できないため、ここで停止
    st.stop()

# --- ページ設定 ---
# Streamlit アプリのページタイトルとレイアウトを設定
st.set_page_config(page_title="データカタログ", layout='wide') # wide レイアウトで表示
# アプリのタイトルを表示
st.title("データカタログアプリ")

# --- 定数 ---
# メタデータ (LLM生成コメント、いいね数など) を格納するテーブル名
METADATA_TABLE_NAME = "DATA_CATALOG_METADATA"
# データベース選択ドロップダウンのデフォルト表示文字列
SELECT_OPTION = "<Select>"
### 選択可能なLLMモデルのリスト定義 ###
AVAILABLE_LLM_MODELS = [
    'claude-3-5-sonnet',
    'llama4-maverick',
    'mistral-large2', 
    'gemma-7b',
    'llama3.1-8b',
    'llama3.1-70b', 
    'snowflake-arctic',
]

# --- メタデータテーブル管理 ---
@st.cache_resource # 結果をキャッシュして、関数呼び出しを高速化 (リソース、ここではテーブル作成状態)
def create_metadata_table():
    """
    データカタログ用のメタデータテーブルが存在しない場合に作成します。
    テーブルの列定義:
    - database_name: データベース名 (主キーの一部)
    - schema_name: スキーマ名 (主キーの一部)
    - table_name: テーブル名 (主キーの一部)
    - table_comment: AIが生成したテーブルの概要説明 (VARCHAR)
    - analysis_ideas: AIが生成した分析アイデア/ユースケース (VARCHAR, JSON形式のリストを想定)
    - embedding: テーブルコメントから生成されたベクトル表現 (VECTOR型)
    - likes: ユーザーによる「いいね」の数 (INTEGER)
    - last_refreshed: このメタデータレコードが最後に更新された日時 (TIMESTAMP_LTZ)
    """
    try:
        # テーブル作成または存在確認のSQL (DDL)
        # VECTOR型の次元数は定数 EMBEDDING_DIMENSION を使用
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE_NAME} (
            database_name VARCHAR(255),
            schema_name VARCHAR(255),
            table_name VARCHAR(255),
            table_comment VARCHAR(16777216), -- LLM生成のテーブル概要 (シンプルな説明)
            analysis_ideas VARCHAR(16777216), -- LLM生成の分析アイデア (ユースケース説明, JSON文字列を想定)
            embedding VECTOR(FLOAT, {EMBEDDING_DIMENSION}), -- ベクトル (次元数はモデル依存)
            likes INT DEFAULT 0, -- いいね数 (デフォルトは0)
            last_refreshed TIMESTAMP_LTZ, -- メタデータ更新日時
            PRIMARY KEY (database_name, schema_name, table_name) -- 複合主キー
        );
        """
        # SQLを実行 (collect()で実行完了を待つ)
        session.sql(ddl).collect()
        # 成功ログを出力
        logger.info(f"{METADATA_TABLE_NAME} テーブルの存在を確認または作成しました。")
        # 成功した場合は True を返す
        return True
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        # 権限不足のエラーメッセージを特定し、より分かりやすく表示
        if "does not exist or not authorized" in str(e):
             st.error(f"エラー: メタデータテーブル '{METADATA_TABLE_NAME}' が存在しないか、アクセス権限がありません。")
             logger.error(f"メタデータテーブルアクセスエラー: {e}")
        # VECTOR型がサポートされていない環境でのエラーハンドリング
        elif "Unsupported feature 'VECTOR'" in str(e):
            st.error("エラー: ご利用のSnowflake環境でVECTOR型がサポートされていません。このアプリは利用できません。")
            logger.error(f"VECTOR型非サポートエラー: {e}")
        # その他のSQLエラー
        else:
             st.error(f"メタデータテーブルの準備中にエラーが発生しました: {e}")
             logger.error(f"{METADATA_TABLE_NAME} テーブルの作成/確認中にエラーが発生しました: {e}")
        # 失敗した場合は False を返す
        return False
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"予期せぬエラー (create_metadata_table): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        # 失敗した場合は False を返す
        return False

# --- データ取得関数 ---

# データベース一覧取得関数
@st.cache_data(ttl=3600) # 結果を1時間キャッシュ
def get_databases():
    """
    アカウント内のデータベース一覧を取得します。
    まず `snowflake.account_usage.databases` ビューからの取得を試みます。
    (権限がない場合やビューが存在しない場合は `SHOW DATABASES` コマンドを使用)
    取得したリストの先頭に選択肢用の '<Select>' を追加して返します。
    account_usage ビューでは、削除済みDBを除外するために DELETED_ON または DELETED カラムを使用します。
    """
    databases_df = None # 取得結果を格納するDataFrame (初期値はNone)
    error_occurred = False # エラー発生フラグ
    error_message = "" # エラーメッセージ格納用

    # account_usage.databases ビューへのアクセス権限を確認
    try:
        # LIMIT 1 で軽いクエリを実行し、ビューが存在しアクセス可能か確認
        session.sql("SELECT 1 FROM snowflake.account_usage.databases LIMIT 1").collect()
        account_usage_accessible = True # アクセス可能
    except SnowparkSQLException as check_err:
        # 権限エラーまたは存在しないエラーの場合
        if "does not exist or not authorized" in str(check_err):
            logger.warning(f"account_usage.databases へのアクセス権限エラー: {check_err}")
            account_usage_accessible = False # アクセス不可
        else:
            # その他のSQLエラーは、後のaccount_usage利用時に処理させる
             account_usage_accessible = True # 利用を試みるフラグは立てておく
             logger.warning(f"account_usage.databases の存在チェックで予期せぬエラー: {check_err}")

    # account_usage がアクセス可能な場合
    if account_usage_accessible:
        try:
            # まず DELETED_ON カラム (比較的新しい) を使って削除済みDBを除外するクエリを試行
            logger.info("account_usage.databases から DELETED_ON を使ってデータベース一覧を取得試行...")
            databases_df = session.sql("""
                SELECT database_name
                FROM snowflake.account_usage.databases
                WHERE deleted_on IS NULL -- 削除日時がNULL（未削除）のDBのみ選択
                ORDER BY database_name -- 名前順でソート
            """).to_pandas() # 結果をPandas DataFrameに変換
            logger.info("DELETED_ON を使った取得に成功。")
        except SnowparkSQLException as e1:
            # DELETED_ON カラムが存在しない場合のエラーハンドリング (古い環境向け)
            if "invalid identifier 'DELETED_ON'" in str(e1):
                logger.warning("DELETED_ON が見つかりません。DELETED カラムを試します。")
                try:
                    # DELETED カラム (古い形式) を使って試行
                    logger.info("account_usage.databases から DELETED を使ってデータベース一覧を取得試行...")
                    databases_df = session.sql("""
                        SELECT database_name
                        FROM snowflake.account_usage.databases
                        WHERE deleted IS NULL -- deleted カラムが NULL (未削除) のDBのみ
                        ORDER BY database_name
                    """).to_pandas()
                    logger.info("DELETED を使った取得に成功。")
                except SnowparkSQLException as e2:
                    # DELETED カラムでも失敗した場合
                    logger.error(f"DELETED を使ったデータベース一覧取得も失敗しました: {e2}")
                    error_occurred = True # エラーフラグを設定
                    error_message = f"account_usage.databases からのデータベース一覧取得に失敗しました (DELETED_ON/DELETED): {e2}"
                except Exception as e_generic_deleted:
                     # DELETED を使う試行中の予期せぬエラー
                     logger.error(f"DELETED を使ったデータベース一覧取得中に予期せぬエラー: {e_generic_deleted}")
                     error_occurred = True
                     error_message = f"account_usage.databases からのデータベース一覧取得中に予期せぬエラー (DELETED): {e_generic_deleted}"
            else:
                # DELETED_ON が存在しないエラー以外のSQLエラー
                logger.error(f"DELETED_ON を使ったデータベース一覧取得中にSQLエラー: {e1}")
                error_occurred = True
                error_message = f"account_usage.databases からのデータベース一覧取得中にSQLエラー: {e1}"
        except Exception as e_generic_deleted_on:
             # DELETED_ON を使う試行中の予期せぬエラー
             logger.error(f"DELETED_ON を使ったデータベース一覧取得中に予期せぬエラー: {e_generic_deleted_on}")
             error_occurred = True
             error_message = f"account_usage.databases からのデータベース一覧取得中に予期せぬエラー (DELETED_ON): {e_generic_deleted_on}"

    # account_usageが使えない、またはエラーが発生した場合、SHOW DATABASES コマンドをフォールバックとして使用
    if databases_df is None or error_occurred:
        # ユーザーにフォールバック使用を通知
        if not account_usage_accessible:
             st.warning("`snowflake.account_usage.databases` へのアクセス権限がないため、`SHOW DATABASES` を使用します。")
        elif error_occurred:
             st.warning(f"account_usageからの取得に失敗したため (`{error_message}`), `SHOW DATABASES` を使用します。")

        try:
            # SHOW DATABASES コマンドを実行
            logger.info("SHOW DATABASES を使ってデータベース一覧を取得試行...")
            databases_show_result = session.sql("SHOW DATABASES").collect() # 結果を行オブジェクトのリストとして取得
            if databases_show_result:
                # 結果が存在する場合
                # 'name' カラムが結果に含まれているか確認 (SHOW DATABASES の標準的な出力形式)
                if databases_show_result[0].__contains__("name"):
                     # 'name' カラムからデータベース名のリストを生成
                     db_names = [row['name'] for row in databases_show_result]
                     # Pandas DataFrame に変換 (account_usage の結果と形式を合わせる)
                     databases_df = pd.DataFrame({'DATABASE_NAME': sorted(db_names)}) # 名前順にソート
                     logger.info("SHOW DATABASES を使った取得に成功。")
                     error_occurred = False # SHOW DATABASESで成功したのでエラーフラグを解除
                else:
                    # 予期しない結果形式の場合
                    logger.error("SHOW DATABASES の結果に 'name' カラムが含まれていません。")
                    error_occurred = True
                    error_message = "SHOW DATABASES の結果形式が予期されたものではありません。"
            else:
                # SHOW DATABASES が結果を返さなかった場合 (DBが全くないなど)
                logger.warning("SHOW DATABASES の結果が空でした。")
                databases_df = pd.DataFrame({'DATABASE_NAME': []}) # 空のDataFrameを作成
                error_occurred = False # 結果が空なのはエラーではない

        except Exception as show_err:
            # SHOW DATABASES の実行自体に失敗した場合
            logger.error(f"SHOW DATABASES の実行に失敗しました: {show_err}")
            # account_usageもSHOW DATABASESも両方失敗した場合のエラーメッセージ
            st.error(f"データベース一覧の取得に失敗しました。account_usageアクセス試行時のエラー: {error_message}, SHOW DATABASES試行時のエラー: {show_err}")
            # 取得不可の場合、デフォルトの選択肢のみ含むリストを返す
            return [SELECT_OPTION]

    # データベース一覧の取得に成功した場合
    if databases_df is not None and not error_occurred:
        # DataFrameからデータベース名のリストを作成
        db_list = databases_df['DATABASE_NAME'].tolist()
        # リストの先頭にデフォルトの選択肢 '<Select>' を追加
        db_list_with_select = [SELECT_OPTION] + db_list
        # ログに取得件数を出力
        logger.info(f"{len(db_list)} 件のデータベースを取得しました。")
        # 結果リストを返す
        return db_list_with_select
    else:
        # フォールバック (SHOW DATABASES) も失敗した場合
        st.error(f"データベース一覧の取得に最終的に失敗しました。エラー: {error_message}")
        # エラー時もデフォルトの選択肢のみ含むリストを返す
        return [SELECT_OPTION]

# 識別子 (DB名, スキーマ名, テーブル名など) の安全性を簡易チェックする関数
def is_safe_identifier(identifier: str) -> bool:
    """
    識別子が安全な文字 (英数字、アンダースコア) で構成されているか簡易的にチェックします。
    SQLインジェクションのリスクを低減するための基本的なチェックです。
    Snowflakeの識別子ルールはより複雑ですが、ここでは危険な文字の有無を確認します。
    Args:
        identifier (str): チェック対象の識別子文字列。
    Returns:
        bool: 安全と判断されれば True、そうでなければ False。
    """
    # 空の識別子は許可しない
    if not identifier:
        return False
    # 危険な可能性のある文字のリスト
    # (セミコロン、コメント開始文字、引用符など)
    forbidden_chars = ['"', "'", ';', '--', '/*', '*/']
    # 識別子内に禁止文字が含まれていないかチェック
    if any(char in identifier for char in forbidden_chars):
        return False
    # Snowflakeでは識別子をダブルクォートで囲むことで、より多くの文字を使用できますが、
    # この関数では、SQLインジェクションに繋がるような基本的な危険文字のみをチェックします。
    # 必要に応じて、正規表現などを用いたより厳密なチェックを追加することも可能です。
    # 例: return bool(re.match(r'^[a-zA-Z_][a-zA-Z0-9_$]*$', identifier)) # 引用符なし識別子のパターン例
    return True # 禁止文字が含まれていなければ True を返す

# 指定されたデータベース内のスキーマ一覧を取得する関数
@st.cache_data(ttl=600) # 結果を10分キャッシュ
def get_schemas_for_database(database_name: str):
    """
    指定されたデータベース内のスキーマ一覧を取得します。
    `INFORMATION_SCHEMA.SCHEMATA` ビューを使用します。
    Args:
        database_name (str): スキーマを取得したいデータベース名。
    Returns:
        list: スキーマ名のリスト。取得失敗時は空リスト。
    """
    # データベース名が選択されていない、またはデフォルト値の場合は空リストを返す
    if not database_name or database_name == SELECT_OPTION:
        return []

    # データベース名の安全性をチェック
    if not is_safe_identifier(database_name):
        st.error(f"不正なデータベース名が指定されました: {database_name}")
        logger.error(f"get_schemas_for_database: 不正なデータベース名 {database_name}")
        return [] # 不正な場合は空リストを返す

    try:
        # クエリテンプレート: 指定データベースの INFORMATION_SCHEMA.SCHEMATA からスキーマ名を取得
        # INFORMATION_SCHEMA と PUBLIC スキーマは除外
        # データベース名を f-string で埋め込み (is_safe_identifier でチェック済みのため安全)
        query = f"""
        SELECT schema_name
        FROM {database_name}.INFORMATION_SCHEMA.SCHEMATA
        WHERE schema_name NOT IN ('INFORMATION_SCHEMA')
        ORDER BY schema_name;
        """
        # SQLを実行し、結果をPandas DataFrameに変換
        schemas_df = session.sql(query).to_pandas()

        # DataFrameからスキーマ名のリストを抽出
        schema_list = schemas_df['SCHEMA_NAME'].tolist()
        # ログに取得件数を出力
        logger.info(f"データベース '{database_name}' から {len(schema_list)} 件のスキーマを取得しました。")
        # スキーマ名のリストを返す
        return schema_list
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        st.warning(f"データベース '{database_name}' のスキーマ取得中にエラーが発生しました: {e}")
        logger.warning(f"get_schemas_for_database エラー ({database_name}): {e}")
        return [] # エラー時は空リストを返す
    except Exception as e:
        # 予期せぬその他のエラー
        st.error(f"スキーマ一覧の取得中に予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"get_schemas_for_database 予期せぬエラー ({database_name}): {e}")
        return [] # エラー時は空リストを返す

# 指定されたデータベース・スキーマ内のテーブル/ビュー一覧を取得する関数
@st.cache_data(ttl=600) # 結果を10分キャッシュ
def get_tables_for_database_schema(database_name: str, selected_schemas: tuple = None): 
    """
    指定されたデータベース内のテーブルとビューの一覧を取得します。
    オプションで特定のスキーマのみに絞り込むことができます。
    `INFORMATION_SCHEMA.TABLES` ビューを使用します。
    Args:
        database_name (str): 対象のデータベース名。
        selected_schemas (tuple, optional): 絞り込むスキーマ名のタプル。Noneの場合は全スキーマ対象。
    Returns:
        pd.DataFrame: テーブル/ビューの情報を含むDataFrame。取得失敗時は空のDataFrame。
    """
    # データベース名が選択されていない、またはデフォルト値の場合は空のDataFrameを返す
    if not database_name or database_name == SELECT_OPTION:
        return pd.DataFrame()

    # データベース名の安全性をチェック
    if not is_safe_identifier(database_name):
        st.error(f"不正なデータベース名が指定されました: {database_name}")
        logger.error(f"get_tables_for_database_schema: 不正なデータベース名 {database_name}")
        return pd.DataFrame()
    # スキーマ名が指定されている場合、それらの安全性もチェック (タプルの各要素をチェック)
    if selected_schemas:
        if not all(is_safe_identifier(s) for s in selected_schemas):
             st.error(f"不正なスキーマ名が含まれています: {selected_schemas}")
             logger.error(f"get_tables_for_database_schema: 不正なスキーマ名 {selected_schemas}")
             return pd.DataFrame()

    try:
        # クエリテンプレート: 指定データベースの INFORMATION_SCHEMA.TABLES から情報を取得
        # 取得するカラム: DB名, スキーマ名, テーブル名, タイプ(TABLE/VIEW), コメント, 行数, サイズ(Bytes), 作成日時, 最終更新日時
        # データベース名はf-stringで埋め込み (チェック済み)
        query = f"""
        SELECT
            TABLE_CATALOG AS DATABASE_NAME,
            TABLE_SCHEMA AS SCHEMA_NAME,
            TABLE_NAME,
            TABLE_TYPE,
            COMMENT AS SOURCE_TABLE_COMMENT, -- 元のテーブルコメント
            ROW_COUNT,
            BYTES,
            CREATED,
            LAST_ALTERED
        FROM {database_name}.INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA != 'INFORMATION_SCHEMA' -- INFORMATION_SCHEMA自体は除外
        """
        # パラメータリスト (スキーマ名用)
        params = []

        # 選択されたスキーマがある場合、WHERE句に条件を追加
        if selected_schemas:
            # スキーマ名の数だけプレースホルダ (?) を生成
            schema_placeholders = ', '.join(['?'] * len(selected_schemas))
            query += f" AND TABLE_SCHEMA IN ({schema_placeholders})" # IN句でスキーマを絞り込み
            params.extend(selected_schemas) # スキーマ名をパラメータリストに追加

        # 結果をスキーマ名、テーブル名でソート
        query += " ORDER BY TABLE_SCHEMA, TABLE_NAME;"

        # SQLを実行 (スキーマ名リストをパラメータとして渡す)
        tables_df = session.sql(query, params=params).to_pandas() # 結果をPandas DataFrameに変換
        # ログ出力用のスキーマ情報文字列を作成
        schema_str = f"スキーマ {selected_schemas}" if selected_schemas else "全スキーマ"
        # ログに取得件数を出力
        logger.info(f"データベース '{database_name}' ({schema_str}) から {len(tables_df)} 件のテーブル/ビューを取得しました。")
        # 結果のDataFrameを返す
        return tables_df

    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        st.warning(f"データベース '{database_name}' のテーブル/ビュー取得中にエラーが発生しました: {e}")
        logger.warning(f"get_tables_for_database_schema エラー ({database_name}, {selected_schemas}): {e}")
        return pd.DataFrame() # エラー時は空のDataFrameを返す
    except Exception as e:
        # 予期せぬその他のエラー
        st.error(f"テーブル/ビュー一覧の取得中に予期せぬエラーが発生しました: {str(e)}")
        logger.error(f"get_tables_for_database_schema 予期せぬエラー ({database_name}, {selected_schemas}): {e}")
        return pd.DataFrame() # エラー時は空のDataFrameを返す

# --- メタデータ取得・更新関数 ---
@st.cache_data(ttl=300) # 結果を5分キャッシュ
def get_metadata(database_name, schema_name, table_name):
    """
    指定されたテーブルのメタデータ (AIコメント、いいね数など) をメタデータテーブルから取得します。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
    Returns:
        dict: メタデータを含む辞書。見つからない場合やエラー時は空の辞書。
    """
    # メタデータテーブル名が設定されていなければ空辞書を返す
    if not METADATA_TABLE_NAME: return {}
    try:
        # メタデータテーブルが存在するかどうかを確認 (より堅牢なチェック)
        try:
            # テーブルオブジェクトの参照を試みる (存在しないか権限がない場合は例外発生)
            session.table(f"{CURRENT_DATABASE}.{CURRENT_SCHEMA}.{METADATA_TABLE_NAME}")
        except SnowparkSQLException:
            # テーブルが見つからない場合
            logger.warning(f"{METADATA_TABLE_NAME} テーブルが見つかりません。メタデータを取得できません。")
            # テーブル作成を試みる
            if not create_metadata_table():
                 # テーブル作成にも失敗した場合
                 st.error(f"メタデータテーブル {METADATA_TABLE_NAME} の準備に失敗しました。")
                 return {} # 空辞書を返す
            # テーブル作成後、再度取得を試みる (ただしキャッシュのため初回は空が返る可能性あり)
            # ここでは初回呼び出しでテーブル作成した場合、今回は空を返す
            return {}

        # メタデータ取得クエリ
        # 指定された DB, Schema, Table に一致するレコードを取得
        query = f"""
        SELECT
            TABLE_COMMENT, -- LLM生成コメント
            ANALYSIS_IDEAS, -- LLM生成分析アイデア (JSON文字列)
            EMBEDDING, -- ベクトルデータ
            LIKES, -- いいね数
            LAST_REFRESHED -- 最終更新日時
        FROM {METADATA_TABLE_NAME}
        WHERE DATABASE_NAME = ? AND SCHEMA_NAME = ? AND TABLE_NAME = ?;
        """
        # SQLを実行し、結果をPandas DataFrameに変換 (パラメータ使用)
        result = session.sql(query, params=[database_name, schema_name, table_name]).to_pandas()

        # 結果が存在するかどうかチェック
        if not result.empty:
            # 結果の最初の行を辞書に変換
            meta = result.iloc[0].to_dict()
            # EMBEDDING カラムの処理: SnowflakeのVECTOR型は通常Pythonリストで返されるが、
            # 文字列で返された場合（古い形式や予期せぬ状況）を考慮し、JSONデコードを試みる
            if 'EMBEDDING' in meta and isinstance(meta['EMBEDDING'], str):
                try:
                    # JSON文字列をPythonリストに変換
                    meta['EMBEDDING'] = json.loads(meta['EMBEDDING'])
                except json.JSONDecodeError:
                    # デコード失敗時はログを出力し、値をNoneに設定
                    logger.warning(f"EMBEDDINGカラムのJSONデコード失敗: {database_name}.{schema_name}.{table_name}")
                    meta['EMBEDDING'] = None
            # メタデータ辞書を返す
            return meta
        else:
            # レコードが見つからなかった場合は空辞書を返す
            return {}
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        logger.warning(f"メタデータ取得中にエラー ({database_name}.{schema_name}.{table_name}): {e}")
        return {} # エラー時は空辞書を返す
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"予期せぬエラー (get_metadata): {e}")
        return {} # エラー時は空辞書を返す


# 指定されたテーブルのメタデータを更新または挿入する関数
def update_metadata(database_name, schema_name, table_name, data_dict):
    """
    指定されたテーブルのメタデータを更新または挿入 (MERGE) します。
    `data_dict` に含まれるキーに基づいて更新/挿入するカラムを動的に決定します。
    VECTOR型のデータは、JSON文字列としてパラメータで渡し、SQL内で `PARSE_JSON` と `CAST` を使用して設定します。
    'LIKES_INCREMENT' キーが True の場合、LIKES カラムを1増やします。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
        data_dict (dict): 更新または挿入するデータを含む辞書。
                          キーはカラム名 (大文字、例: 'TABLE_COMMENT', 'ANALYSIS_IDEAS', 'EMBEDDING', 'LIKES')
                          または特殊キー 'LIKES_INCREMENT'。
    Returns:
        bool: 処理が成功した場合は True、失敗した場合は False。
    """
    # メタデータテーブル名が設定されていなければ処理中断
    if not METADATA_TABLE_NAME: return False
    try:
        # MERGE文の UPDATE SET 句のリスト
        update_clauses = []
        # MERGE文の INSERT 時のカラムリスト
        insert_cols = ["database_name", "schema_name", "table_name", "last_refreshed"]
        # MERGE文の INSERT 時のVALUESリスト (sourceを参照)
        insert_vals = ["source.database_name", "source.schema_name", "source.table_name", "CURRENT_TIMESTAMP()"]
        # MERGE文の USING句の SELECT リスト (パラメータ用)
        source_cols = ["? AS database_name", "? AS schema_name", "? AS table_name"]
        # 基本となるパラメータ (DB名, スキーマ名, テーブル名)
        params_base = [database_name, schema_name, table_name]
        # data_dict の内容に応じて動的に追加されるパラメータリスト
        params_dynamic = []
        # Embedding 用の JSON 文字列パラメータ (存在する場合のみ設定)
        embedding_param_json = None

        # --- data_dict の各キーに基づいて SQL句とパラメータを構築 ---

        # 'TABLE_COMMENT' が data_dict に含まれる場合
        if 'TABLE_COMMENT' in data_dict:
            update_clauses.append("table_comment = source.table_comment") # UPDATE句
            insert_cols.append("table_comment")                         # INSERTカラム
            insert_vals.append("source.table_comment")                  # INSERT値
            source_cols.append("? AS table_comment")                    # USING句
            params_dynamic.append(data_dict['TABLE_COMMENT'])           # パラメータ

        # 'ANALYSIS_IDEAS' が data_dict に含まれる場合
        if 'ANALYSIS_IDEAS' in data_dict:
            update_clauses.append("analysis_ideas = source.analysis_ideas") # UPDATE句
            insert_cols.append("analysis_ideas")                       # INSERTカラム
            insert_vals.append("source.analysis_ideas")                # INSERT値
            source_cols.append("? AS analysis_ideas")                  # USING句
            params_dynamic.append(data_dict['ANALYSIS_IDEAS'])         # パラメータ (JSON文字列を想定)

        # 'EMBEDDING' が data_dict に含まれる場合
        if 'EMBEDDING' in data_dict:
            embedding_list = data_dict['EMBEDDING']
            # VECTOR型にキャストするSQL式 (パラメータプレースホルダを使用)
            # PARSE_JSON(?) でパラメータのJSON文字列をVARIANTに変換し、::VECTOR(...) でキャスト
            embedding_sql_expr = f"(PARSE_JSON(?))::VECTOR(FLOAT, {EMBEDDING_DIMENSION})"

            # embedding_list が有効なリストの場合
            if embedding_list is not None and isinstance(embedding_list, list):
                 # UPDATE句とINSERT句に VECTOR型キャスト式を追加
                 update_clauses.append(f"embedding = {embedding_sql_expr}")
                 insert_cols.append("embedding")
                 insert_vals.append(embedding_sql_expr)
                 # パラメータとして渡すためのJSON文字列を生成
                 embedding_param_json = json.dumps(embedding_list)
            else:
                 # Embedding が None またはリストでない場合は NULL を設定
                 update_clauses.append("embedding = NULL")
                 # INSERT 時は embedding 列を含めない (テーブル定義のデフォルト=NULLが使われる)
                 logger.warning(f"Embedding for {table_name} is None or not a list, setting to NULL.")

        # 'LIKES_INCREMENT' が data_dict に含まれ、True の場合 (いいねボタン用)
        if 'LIKES_INCREMENT' in data_dict and data_dict['LIKES_INCREMENT']:
            # UPDATE句で既存の likes 値に 1 を加算 (target.likes を参照)
            update_clauses.append("likes = NVL(target.likes, 0) + 1") # 既存値がNULLの場合も考慮
            # INSERT 時には LIKES_INCREMENT は影響しない (新規作成ならデフォルトの0)
        # 'LIKES' が data_dict に含まれる場合 (直接値を設定する場合、通常は使わないかも)
        elif 'LIKES' in data_dict:
             update_clauses.append("likes = source.likes") # UPDATE句
             insert_cols.append("likes")                  # INSERTカラム
             insert_vals.append("source.likes")           # INSERT値
             source_cols.append("? AS likes")             # USING句
             params_dynamic.append(data_dict['LIKES'])    # パラメータ

        # --- MERGE 文の構築 ---
        # 常に last_refreshed は現在時刻で更新する
        update_clauses.append("last_refreshed = CURRENT_TIMESTAMP()")

        # MERGE文のテンプレート
        merge_sql = f"""
        MERGE INTO {METADATA_TABLE_NAME} AS target -- 対象テーブル
        USING (SELECT {', '.join(source_cols)}) AS source -- 更新/挿入データソース (パラメータから生成)
        ON target.database_name = source.database_name -- マッチング条件 (主キー)
           AND target.schema_name = source.schema_name
           AND target.table_name = source.table_name
        WHEN MATCHED THEN -- マッチした場合 (UPDATE)
            UPDATE SET {', '.join(update_clauses)} -- update_clauses リストを展開
        WHEN NOT MATCHED THEN -- マッチしなかった場合 (INSERT)
            INSERT ({', '.join(insert_cols)}) -- insert_cols リストを展開
            VALUES ({', '.join(insert_vals)}); -- insert_vals リストを展開
        """

        # --- パラメータリストの最終化 ---
        # 基本パラメータ + 動的パラメータ
        final_params = params_base + params_dynamic
        # SQL文中のプレースホルダ (?) の数をカウント
        param_count_in_sql = merge_sql.count('?')

        # Embedding パラメータが必要な回数だけ追加されるように調整
        # embedding_sql_expr (PARSE_JSON(?) を含む式) が UPDATE と INSERT の両方で使われる可能性があるため
        expected_embedding_params = 0
        if embedding_param_json is not None:
            # UPDATE句で embedding_sql_expr が使われているかチェック
            if f"embedding = {embedding_sql_expr}" in merge_sql:
                expected_embedding_params += 1
            # INSERTのVALUES句で embedding_sql_expr が使われているかチェック
            if embedding_sql_expr in insert_vals:
                expected_embedding_params += 1

            # 必要な回数だけ embedding_param_json をパラメータリストに追加
            for _ in range(expected_embedding_params):
                final_params.append(embedding_param_json)

        # 最終的なパラメータ数とSQL内のプレースホルダ数が一致するか検証 (デバッグ用)
        if len(final_params) != param_count_in_sql:
             logger.error(f"Parameter count mismatch! SQL needs {param_count_in_sql}, but got {len(final_params)}.")
             logger.error(f"SQL: {merge_sql}")
             logger.error(f"Params: {final_params}") # パラメータ内容もログ出力 (機密情報に注意)
             st.error("内部エラー: メタデータ更新時のパラメータ数が一致しません。ログを確認してください。")
             return False # 不一致の場合はエラーとして処理中断


        # --- SQL 実行 ---
        # 実行するSQLとパラメータをデバッグログに出力
        logger.debug(f"Executing MERGE SQL for {table_name}: {merge_sql}")
        logger.debug(f"With Params ({len(final_params)} items): {final_params}")
        # MERGE文を実行 (パラメータを渡す)
        session.sql(merge_sql, params=final_params).collect()

        # 成功ログを出力
        logger.info(f"メタデータを更新/挿入しました: {database_name}.{schema_name}.{table_name}, updated_keys: {list(data_dict.keys())}")
        # 関連するキャッシュをクリア (重要: get_metadata や get_all_metadata のキャッシュを無効化)
        st.cache_data.clear()
        # 成功した場合は True を返す
        return True

    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        logger.error(f"メタデータ更新中にSQLエラー ({table_name}): {e}", exc_info=True) # トレースバックもログに出力
        # 失敗したSQLとパラメータをログに出力（デバッグのため）
        try:
            # final_params が定義されているか確認
            log_params = final_params if 'final_params' in locals() else params_base + params_dynamic
            logger.error(f"Failed SQL: {merge_sql}") # merge_sql は try ブロック内で定義されているはず
            logger.error(f"Failed Params: {log_params}") # パラメータ内容 (機密情報に注意)
        except NameError:
             logger.error("Failed to log SQL/Params due to NameError.") # 変数が未定義の場合のエラー

        # Streamlit UIにエラーメッセージを表示
        st.error(f"メタデータ更新中にSQLエラー ({table_name})。ログを確認してください。エラー: {e}")
        # 失敗した場合は False を返す
        return False
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"メタデータ更新中に予期せぬエラー ({table_name}): {e}", exc_info=True) # トレースバックもログに出力
        st.error(f"メタデータ更新中に予期せぬエラー ({table_name}) が発生しました: {e}")
        # 失敗した場合は False を返す
        return False


# メタデータテーブルから全てのメタデータを取得する関数
@st.cache_data(ttl=300) # 結果を5分キャッシュ
def get_all_metadata():
    """
    メタデータテーブル (`DATA_CATALOG_METADATA`) から全てのメタデータレコードを取得します。
    Returns:
        dict: テーブルの完全修飾名 (DB.SCHEMA.TABLE) をキーとし、
              メタデータ辞書を値とする辞書。エラー時は空の辞書。
    """
    # メタデータテーブル名が設定されていなければ空辞書を返す
    if not METADATA_TABLE_NAME: return {}
    try:
        # メタデータテーブルが存在するかどうかを確認
        try:
            session.table(f"{CURRENT_DATABASE}.{CURRENT_SCHEMA}.{METADATA_TABLE_NAME}")
        except SnowparkSQLException:
            # テーブルが見つからない場合
            logger.warning(f"{METADATA_TABLE_NAME} テーブルが見つかりません。メタデータを取得できません。")
            # テーブル作成を試みる
            if not create_metadata_table():
                 st.error(f"メタデータテーブル {METADATA_TABLE_NAME} の準備に失敗しました。")
                 return {}
            # テーブル作成直後はデータがないので空辞書を返す
            return {}

        # 全てのレコードを取得するクエリ
        query = f"SELECT * FROM {METADATA_TABLE_NAME};"
        # SQLを実行し、結果をPandas DataFrameに変換
        metadata_df = session.sql(query).to_pandas()
        # 結果を格納するための空の辞書を初期化
        metadata_dict = {}
        # DataFrameの各行を処理
        for _, row in metadata_df.iterrows():
            # テーブルの完全修飾名をキーとして生成
            key = f"{row['DATABASE_NAME']}.{row['SCHEMA_NAME']}.{row['TABLE_NAME']}"
            # 行データを辞書に変換
            meta = row.to_dict()
            # Embedding カラムの処理 (get_metadata と同様)
            if 'EMBEDDING' in meta and isinstance(meta['EMBEDDING'], str):
                try:
                    meta['EMBEDDING'] = json.loads(meta['EMBEDDING'])
                except json.JSONDecodeError:
                    logger.warning(f"get_all_metadata: EMBEDDINGカラムのJSONデコード失敗: {key}")
                    meta['EMBEDDING'] = None
            # キーとメタデータ辞書のペアを結果辞書に追加
            metadata_dict[key] = meta
        # ログに取得件数を出力
        logger.info(f"{len(metadata_dict)} 件のメタデータを取得しました。")
        # 結果の辞書を返す
        return metadata_dict
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        logger.error(f"全メタデータ取得中にエラー: {e}")
        st.error(f"メタデータの取得中にエラーが発生しました: {e}")
        return {} # エラー時は空辞書を返す
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"予期せぬエラー (get_all_metadata): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        return {} # エラー時は空辞書を返す

# --- アクセス数取得関数 (新規追加) ---
@st.cache_data(ttl=3600) # 結果を1時間キャッシュ
def get_monthly_access_count(database_name, schema_name, table_name):
    """
    指定されたテーブルに対する直近1ヶ月間のアクセス数を取得します。
    `snowflake.account_usage.access_history` ビューを使用します。
    このビューへのアクセス権限が必要です。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
    Returns:
        int or str: アクセス数 (int)。権限がない場合は "N/A"。エラー時は "エラー"。
                     0回の場合は 0 (int)。
    """
    try:
        # 検索対象期間の開始日時 (UTCで現在から30日前) を計算
        one_month_ago = (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        # テーブルの完全修飾名を作成 (大文字に統一、Snowflakeの識別子は通常大文字で格納されるため)
        full_table_name = f"{database_name}.{schema_name}.{table_name}".upper()

        # access_history ビューへのアクセス権限を確認
        try:
            session.sql("SELECT 1 FROM snowflake.account_usage.access_history LIMIT 1").collect()
        except SnowparkSQLException as check_err:
            # 権限エラーまたは存在しないエラーの場合
            if "does not exist or not authorized" in str(check_err):
                logger.warning(f"snowflake.account_usage.access_history へのアクセス権限がないため、アクセス数を取得できません。")
                # 権限がない場合は "N/A" を返す
                return "N/A"
            else:
                # その他のSQLエラーの場合は、そのまま上位のexceptブロックで処理させる
                raise check_err

        # アクセス数をカウントするクエリ
        # query_start_time で期間を絞り込み
        # direct_objects_accessed (直接参照されたオブジェクト) または
        # base_objects_accessed (ビューなどの基底オブジェクトとして参照された) のいずれかに
        # 対象テーブルの完全修飾名が含まれるクエリの数をカウント
        # 注意: このクエリは大規模な環境ではコストがかかる可能性があります。
        #       パフォーマンス改善のためには、対象テーブルを絞り込むなどの工夫が必要な場合があります。
        query = f"""
        SELECT COUNT(*) as access_count
        FROM snowflake.account_usage.access_history
        WHERE query_start_time >= '{one_month_ago}' -- 期間指定
        AND (
             -- 配列内に指定したテーブル名が含まれるかチェック (VARIANT型にキャスト)
             ARRAY_CONTAINS('{full_table_name}'::variant, direct_objects_accessed)
             OR
             ARRAY_CONTAINS('{full_table_name}'::variant, base_objects_accessed)
        );
        """

        # SQLを実行し、結果を取得
        result = session.sql(query).collect()
        # 結果が存在する場合
        if result:
            # 最初の行から 'ACCESS_COUNT' カラムの値を取得
            count = result[0]['ACCESS_COUNT']
            # デバッグログに取得結果を出力
            logger.debug(f"アクセス数取得成功 ({full_table_name}): {count}")
            # カウント数を返す
            return count
        else:
            # 結果が空の場合 (通常は COUNT(*) なので0が返るはずだが念のため)
            logger.warning(f"アクセス数の取得結果が空でした ({full_table_name})")
            return 0 # 0を返す
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        logger.error(f"アクセス数取得中にSQLエラー ({database_name}.{schema_name}.{table_name}): {e}")
        # エラーが発生したことを示す文字列を返す
        return "エラー"
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"アクセス数取得中に予期せぬエラー ({database_name}.{schema_name}.{table_name}): {e}")
        # エラーが発生したことを示す文字列を返す
        return "エラー"


# --- LLM連携関数 ---
@st.cache_data(ttl=3600) # 結果を1時間キャッシュ
def get_table_schema(database_name, schema_name, table_name):
    """
    指定されたテーブルのカラム情報を `INFORMATION_SCHEMA.COLUMNS` から取得します。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
    Returns:
        pd.DataFrame: カラム情報 (COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT) を含むDataFrame。
                      取得失敗時は空のDataFrame。
    """
    # データベース名、スキーマ名、テーブル名の安全性をチェック
    if not is_safe_identifier(database_name) or \
       not is_safe_identifier(schema_name) or \
       not is_safe_identifier(table_name):
        st.error(f"不正な識別子が含まれています: DB={database_name}, SC={schema_name}, TBL={table_name}")
        logger.error(f"get_table_schema: 不正な識別子 DB={database_name}, SC={schema_name}, TBL={table_name}")
        return pd.DataFrame() # 不正な場合は空のDataFrameを返す

    try:
        # クエリテンプレート: 指定DBの INFORMATION_SCHEMA.COLUMNS から情報を取得
        # 取得カラム: カラム名, データ型, NULL許容か, カラムコメント
        # データベース名はf-stringで埋め込み (チェック済み)
        # スキーマ名とテーブル名はパラメータ (?) で指定
        full_query = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COMMENT
        FROM {database_name}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
        ORDER BY ORDINAL_POSITION; -- カラムの定義順でソート
        """
        # パラメータリスト (スキーマ名, テーブル名)
        params = [schema_name, table_name]
        # SQLを実行し、結果をPandas DataFrameに変換
        schema_df = session.sql(full_query, params=params).to_pandas()

        # 結果が空かどうかチェック
        if schema_df.empty:
             # スキーマ情報が見つからなかった場合 (テーブルが存在しないか、カラムがない)
             logger.warning(f"スキーマ情報が見つかりません: {database_name}.{schema_name}.{table_name}")
        else:
             # 取得成功ログ
             logger.info(f"スキーマ情報を取得: {database_name}.{schema_name}.{table_name}")
        # 結果のDataFrameを返す (空の場合も含む)
        return schema_df
    except SnowparkSQLException as e:
        # SQL実行中にエラーが発生した場合
        # テーブルが存在しない、または権限がない場合のエラーハンドリング
        if "does not exist or not authorized" in str(e):
             logger.warning(f"テーブルが見つからないか権限がありません: {database_name}.{schema_name}.{table_name}")
             # この場合、UIにエラーを出すと冗長になる可能性があるため、ログのみとする
        else:
             # その他のSQLエラー
             logger.error(f"スキーマ情報取得エラー ({database_name}.{schema_name}.{table_name}): {e}")
             # UIには警告として表示 (エラーにするとアプリの動作に影響する場合があるため)
             st.warning(f"テーブル '{database_name}.{schema_name}.{table_name}' のスキーマ情報取得中にエラーが発生しました: {e}")
        # エラー時も空のDataFrameを返す (LLM生成処理などで後続処理が継続できるように)
        return pd.DataFrame()
    except Exception as e:
        # 予期せぬその他のエラー
        logger.error(f"予期せぬエラー (get_table_schema): {e}")
        st.error(f"予期せぬエラーが発生しました: {e}")
        return pd.DataFrame() # エラー時も空のDataFrameを返す

# LLMを使用してテーブルコメントと分析アイデアを生成する関数
def generate_comment_and_ideas(database_name, schema_name, table_name, source_table_comment, model=DEFAULT_LLM_MODEL): # model引数はそのまま
    """
    Snowflake Cortex LLM 関数 (`SNOWFLAKE.CORTEX.COMPLETE`) を使用して、
    テーブルスキーマと既存コメントに基づき、簡潔なテーブルコメントと分析アイデアを生成します。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
        source_table_comment (str or None): INFORMATION_SCHEMA から取得した元のテーブルコメント。
        model (str, optional): 使用するLLMモデル名。デフォルトは `DEFAULT_LLM_MODEL`。
    Returns:
        tuple (str or None, list or None): (生成されたコメント, 分析アイデアのリスト)。
                                            エラー発生時は (None, None) またはエラーを示す文字列/リスト。
    """
    try:
        # まず、対象テーブルのスキーマ情報を取得
        schema_df = get_table_schema(database_name, schema_name, table_name)
        # スキーマ情報が取得できなかった場合 (テーブルが存在しない、権限がない、エラーなど)
        if schema_df.empty:
            # ログに警告を出力し、LLM生成をスキップ
            logger.warning(f"スキーマ情報が取得できなかったため、LLM生成をスキップします: {database_name}.{schema_name}.{table_name}")
            # UIに警告を出すと大量に表示される可能性があるのでログのみ
            # (呼び出し元でエラーハンドリングが必要)
            return None, None # None, None を返してスキップを示す

        # スキーマ情報をテキスト形式に整形 (LLMへの入力用)
        schema_text = "カラム名 | データ型 | NULL許容 | カラムコメント\n------- | -------- | -------- | --------\n"
        for _, row in schema_df.iterrows():
            # 各カラムの情報を取得 (欠損値は空文字で代替)
            col_name = row['COLUMN_NAME'] if pd.notna(row['COLUMN_NAME']) else ""
            data_type = row['DATA_TYPE'] if pd.notna(row['DATA_TYPE']) else ""
            nullable = row['IS_NULLABLE'] if pd.notna(row['IS_NULLABLE']) else ""
            # カラムコメントは None や NaN の場合に備えて文字列に変換
            comment_str = str(row['COMMENT']) if pd.notna(row['COMMENT']) and row['COMMENT'] else ""
            # Markdown テーブル形式で追記
            schema_text += f"{col_name} | {data_type} | {nullable} | {comment_str}\n"

        # 元のテーブルコメントが存在すれば、プロンプトに追加
        source_comment_text = f"\n既存のテーブルコメント: {source_table_comment}" if source_table_comment and pd.notna(source_table_comment) else ""

        # LLMへの指示 (プロンプト) を作成
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
        # Snowflake Cortex Complete関数を呼び出すSQL
        # 第一引数にモデル名、第二引数にプロンプトを渡す
        sql_query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)"
        # SQLに渡すパラメータリスト (model引数を使用)
        params = [model, prompt]

        try:
            # LLM呼び出し開始のログ (使用モデル名も出力)
            logger.info(f"Cortex Complete呼び出し開始 ({database_name}.{schema_name}.{table_name}), model={model}") # SQLクエリは冗長なので省略
            # SQLを実行し、結果を取得 (collect() は通常リストを返す)
            response = session.sql(sql_query, params=params).collect()
            # LLM呼び出し完了のログ
            logger.info(f"Cortex Complete呼び出し完了 ({database_name}.{schema_name}.{table_name}), model={model}")

            # 結果の処理 (response は [Row(SNOWFLAKE.CORTEX.COMPLETE(?, ?)=result_string)] のような形)
            if response and response[0] and response[0][0]:
                # 最初の行の最初の列にある応答文字列を取得
                result_json_str = response[0][0]

                # --- LLM応答からJSON部分を抽出する処理 ---
                # 応答がマークダウンのコードブロック (```json ... ```) で囲まれている場合や、
                # JSONオブジェクトが直接返される場合に対応
                try:
                    # まず ```json ... ``` 形式を探す
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```|(\{.*?\})', result_json_str, re.DOTALL | re.IGNORECASE)
                    if json_match:
                        # マッチしたグループの中から中身がある方 (JSON部分) を取得
                        result_json_str = next(g for g in json_match.groups() if g is not None)
                    else:
                        # ```json ... ``` がない場合、最初と最後の波括弧を探して抽出を試みる
                        start_index = result_json_str.find('{')
                        end_index = result_json_str.rfind('}')
                        if start_index != -1 and end_index != -1 and start_index < end_index:
                            result_json_str = result_json_str[start_index:end_index+1]
                        else:
                            # JSONの開始/終了が見つからない場合はエラー
                            raise ValueError("LLM応答からJSON部分を抽出できませんでした。")
                except (ValueError, AttributeError) as extract_err:
                    # JSON抽出に失敗した場合
                    logger.error(f"LLM応答からJSON部分の抽出に失敗 ({database_name}.{schema_name}.{table_name}, model={model}): {extract_err}. Raw response: {result_json_str[:500]}...")
                    st.error(f"LLM({model})からの応答形式が不正です。JSON部分を抽出できませんでした。")
                    # エラーを示す値を返す
                    return "AIコメント生成失敗(応答抽出エラー)", ["AI分析アイデア生成失敗(応答抽出エラー)"]

                # 抽出したJSON文字列をログに出力 (長すぎる場合は最初の200文字)
                logger.info(f"LLM応答 (抽出後JSON) ({database_name}.{schema_name}.{table_name}, model={model}): {result_json_str[:200]}...")
                # --- JSON文字列をパースしてデータを取得 ---
                try:
                    # JSON文字列をPython辞書に変換
                    result_data = json.loads(result_json_str)
                    # 'table_comment' キーの値を取得
                    generated_comment = result_data.get("table_comment")
                    # 'analysis_ideas' キーの値 (リストのはず) を取得
                    ideas = result_data.get("analysis_ideas")

                    # --- 取得した値の検証 ---
                    # コメントが取得できたか、文字列か、空でないかチェック
                    if not generated_comment or not isinstance(generated_comment, str) or not generated_comment.strip():
                        logger.warning(f"LLM応答からtable_commentが取得/検証できませんでした (model={model})。応答: {result_json_str}")
                        generated_comment = "AIコメント生成失敗(不正な形式)" # 失敗を示す文字列
                    # アイデアがリスト形式か、空でないかチェック
                    if not isinstance(ideas, list) or not ideas:
                        logger.warning(f"LLM応答のanalysis_ideasがリスト形式でないか空です (model={model})。応答: {result_json_str}")
                        ideas = ["AI分析アイデア生成失敗(不正な形式)"] # 失敗を示すリスト
                    # アイデアリスト内の各要素が空でない文字列かチェック
                    elif not all(isinstance(idea, str) and idea.strip() for idea in ideas):
                        logger.warning(f"LLM応答のanalysis_ideasに不正な要素が含まれます (model={model})。応答: {result_json_str}")
                        # 不正な要素を除外し、有効なものだけ残す (空なら失敗リストに)
                        ideas = [idea for idea in ideas if isinstance(idea, str) and idea.strip()]
                        if not ideas: ideas = ["AI分析アイデア生成失敗(不正な要素)"]

                    # 検証済みのコメントとアイデアリストを返す
                    return generated_comment, ideas
                except json.JSONDecodeError:
                    # JSONパース自体に失敗した場合
                    logger.error(f"LLM応答のJSONパースに失敗 ({database_name}.{schema_name}.{table_name}, model={model}): {result_json_str}")
                    st.error(f"LLM({model})からの応答の解析に失敗しました。")
                    return "AIコメント生成失敗(JSONパースエラー)", ["AI分析アイデア生成失敗(JSONパースエラー)"]
                except Exception as parse_err:
                    # その他のパース/処理中のエラー
                    logger.error(f"LLM応答の処理中にエラー ({database_name}.{schema_name}.{table_name}, model={model}): {parse_err}")
                    st.error(f"LLM({model})応答の処理中に予期せぬエラーが発生しました。")
                    return "AIコメント生成失敗(処理エラー)", ["AI分析アイデア生成失敗(処理エラー)"]
            else:
                # LLMからの応答が空だった場合
                logger.error(f"LLMからの応答が空です ({database_name}.{schema_name}.{table_name}, model={model})")
                st.error(f"LLM({model})からの応答がありませんでした。")
                # 失敗を示す None を返す
                return None, None
        except SnowparkSQLException as e:
            # LLM呼び出し (session.sql) でSQLエラーが発生した場合
            # エラーメッセージとクエリ、パラメータ(プロンプトは長いため<prompt>で代替)をログに出力
            logger.error(f"LLM呼び出しエラー ({database_name}.{schema_name}.{table_name}, model={model}): {e}", exc_info=True) # SQLクエリはログ出力省略推奨
            # モデルが見つからない場合のエラーメッセージをより具体的に
            if "Invalid argument" in str(e) and model in str(e):
                st.error(f"LLM (Complete) 呼び出しエラー: モデル '{model}' が無効または利用できません。エラー: {e}")
            else:
                st.error(f"LLM (Complete) 呼び出し中にSQLエラーが発生しました (model: {model})。ログを確認してください。エラー: {e}")
            # 失敗を示す None を返す
            return None, None
    except Exception as e:
        # この関数内のその他の予期せぬエラー
        logger.error(f"予期せぬエラー (generate_comment_and_ideas, model={model}): {e}", exc_info=True) # トレースバックもログに
        st.error(f"予期せぬエラーが発生しました: {e}")
        # 失敗を示す None を返す
        return None, None



# テキストのベクトル表現 (Embedding) を生成する関数
def generate_embedding(text, model=DEFAULT_EMBEDDING_MODEL):
    """
    Snowflake Cortex Embed Text 関数 (`SNOWFLAKE.CORTEX.EMBED_TEXT_*`) を使用して、
    与えられたテキストのベクトル表現 (Embedding) を生成します。
    Args:
        text (str): ベクトル化するテキスト。
        model (str, optional): 使用する埋め込みモデル名。デフォルトは `DEFAULT_EMBEDDING_MODEL`。
    Returns:
        list or None: 生成されたベクトル (float のリスト)。エラーの場合は None。
    """
    # 入力テキストが有効か (空でない文字列か) チェック
    if not text or not isinstance(text, str) or not text.strip():
        logger.warning("ベクトル生成のための有効なテキストがありません。")
        return None # 無効な場合は None を返す

    # Snowflake Cortex Embed Text 関数を呼び出す SQL
    # 第一引数にモデル名、第二引数にテキストを渡す
    # 関数名は EMBED_FUNCTION_NAME 定数を使用 (次元数を含む)
    sql_query = f"SELECT {EMBED_FUNCTION_NAME}(?, ?)"
    # SQLに渡すパラメータリスト
    params = [model, text]

    try:
        # ベクトル生成呼び出し開始のログ (モデル名、関数名)
        logger.info(f"Cortex Embed呼び出し開始 (model={model}, func={EMBED_FUNCTION_NAME})") # SQLクエリ省略
        # SQLを実行し、結果を取得 (collect() はリストを返す)
        result_df = session.sql(sql_query, params=params).collect()
        # ベクトル生成呼び出し完了のログ
        logger.info(f"Cortex Embed呼び出し完了")

        # 結果の処理 (response は [Row(EMBED_TEXT...(...)=vector_list)] のような形)
        if result_df and result_df[0] and result_df[0][0]:
            # 最初の行の最初の列にある結果 (ベクトルのはず) を取得
            embedding_vector = result_df[0][0]

            # --- ベクトル形式の検証 ---
            # まれに文字列で返る可能性も考慮 (古い形式など)
            if isinstance(embedding_vector, str):
                try:
                    # JSONデコードを試みる
                    embedding_vector = json.loads(embedding_vector)
                except json.JSONDecodeError as e:
                    # デコード失敗時
                    logger.error(f"ベクトル結果のJSONデコードエラー: {e}, 元の文字列: {embedding_vector[:100]}...")
                    st.error("ベクトル生成結果の解析に失敗しました。")
                    return None

            # 結果が期待する形式 (リスト) か、次元数が正しいかチェック
            if isinstance(embedding_vector, list) and len(embedding_vector) == EMBEDDING_DIMENSION:
                # 成功ログを出力
                logger.info(f"テキストのベクトル生成に成功しました (次元数: {len(embedding_vector)})")
                # 生成されたベクトルリストを返す
                return embedding_vector
            else:
                 # 形式または次元数が不正な場合
                 logger.error(f"生成されたベクトルの形式または次元数が不正です。型: {type(embedding_vector)}, 次元数: {len(embedding_vector) if isinstance(embedding_vector, list) else 'N/A'}, 期待値: {EMBEDDING_DIMENSION}")
                 st.error(f"生成されたベクトルの形式/次元数が不正です (期待値: {EMBEDDING_DIMENSION})。")
                 return None # 失敗として None を返す
        else:
            # ベクトル生成の結果が空だった場合
            logger.error("ベクトル生成の結果が空です。")
            st.error("ベクトル生成の結果がありませんでした。")
            return None # 失敗として None を返す
    except SnowparkSQLException as e:
        # ベクトル生成 (session.sql) でSQLエラーが発生した場合
        logger.error(f"ベクトル生成中にSnowpark SQLエラーが発生しました (model={model}, func={EMBED_FUNCTION_NAME}): {e}", exc_info=True) # SQLクエリ省略
        # エラーの種類に応じて、より具体的なメッセージをUIに表示
        # 関数が見つからない、または権限がないエラー
        if "invalid identifier" in str(e) or "does not exist or not authorized" in str(e):
             logger.error(f"Cortex関数 {EMBED_FUNCTION_NAME} の呼び出しエラー: {e}")
             st.error(f"ベクトル生成関数({EMBED_FUNCTION_NAME})が見つからないか、権限がありません。")
        # 指定されたモデルが存在しないエラー (エラーメッセージは環境による可能性あり)
        elif "Invalid argument" in str(e) and model in str(e):
            logger.error(f"指定されたモデル '{model}' が見つかりません: {e}")
            st.error(f"ベクトル生成モデル '{model}' が無効または利用できません。エラー: {e}")
        # その他のSQLエラー
        else:
            st.error(f"ベクトル生成中にSQLエラーが発生しました。ログを確認してください。エラー: {e}")
        # 失敗として None を返す
        return None
    except Exception as e:
        # この関数内のその他の予期せぬエラー
        logger.error(f"ベクトル生成中に予期せぬエラーが発生しました: {e}", exc_info=True)
        st.error(f"ベクトル生成中に予期せぬエラーが発生しました: {e}")
        # 失敗として None を返す
        return None


# LLMによるメタデータ (コメント、アイデア、ベクトル) を生成し、保存する関数
def generate_and_save_ai_metadata(database_name, schema_name, table_name, source_table_comment, model=DEFAULT_LLM_MODEL):
    """
    指定されたテーブルについて、以下の処理を実行します:
    1. LLMでテーブルコメントと分析アイデアを生成 (`generate_comment_and_ideas`) - 指定されたモデルを使用
    2. 生成されたコメントに基づいてベクトル表現を生成 (`generate_embedding`)
    3. 生成されたメタデータをメタデータテーブルに保存 (`update_metadata`)
    処理の進行状況を Streamlit のスピナーで表示します。
    Args:
        database_name (str): データベース名。
        schema_name (str): スキーマ名。
        table_name (str): テーブル名。
        source_table_comment (str or None): 元のテーブルコメント。
        model (str): 使用するLLMモデル名。
    Returns:
        bool: 全ての処理が成功した場合は True、いずれかで失敗した場合は False。
    """
    # 処理中のメッセージを表示するためのスピナーウィジェット
    progress_text = f"{database_name}.{schema_name}.{table_name}: AIコメントと分析アイデアを生成中 (モデル: {model})..."
    # st.spinner を使うと with ブロック終了時に自動で消える
    with st.spinner(progress_text):

        # 1. コメントとアイデアを生成 (model 引数を渡す) 
        generated_comment, ideas = generate_comment_and_ideas(database_name, schema_name, table_name, source_table_comment, model=model)

        # コメントまたはアイデアの生成に失敗した場合 (戻り値が None)
        if generated_comment is None or ideas is None:
            # スピナーは with ブロック終了で消えるのでここでは何もしない
            # UIにエラーメッセージを表示 (generate_comment_and_ideas 内で表示されているはずだが念のため)
            st.error(f"'{table_name}' のAIコメントまたは分析アイデアの生成に失敗しました (モデル: {model})。")
            return False # 失敗として False を返す
        # コメント生成には失敗したが、アイデアは取得できた場合 (エラー文字列が入っている)
        if "生成失敗" in str(generated_comment):
            # UIに警告を表示
            st.warning(f"'{table_name}' のAIコメント生成に失敗しました (モデル: {model})。分析アイデアのみ保存を試みます。")
            # この場合でも、アイデアの保存とベクトル生成(空コメントベースなので失敗するが)に進む

        # 2. ベクトルを生成 (有効なコメントが生成された場合のみ)
        embedding = None # 初期値は None
        # コメントが生成され、かつエラー文字列でない場合
        if generated_comment and "生成失敗" not in str(generated_comment):
            # スピナー内のテキストを更新 (with spinner ブロック内なので spinner オブジェクトは不要)
            # st.spinner のテキストは更新できないため、ログで代替
            logger.info(f"{database_name}.{schema_name}.{table_name}: ベクトルを生成中...")
            # ベクトル生成関数を呼び出し
            embedding = generate_embedding(generated_comment) # Embeddingモデルは固定

            # ベクトル生成に失敗した場合
            if embedding is None:
                # UIに警告を表示
                st.warning(f"'{table_name}' のベクトル生成に失敗しました。コメントとアイデアのみ保存します。")
        else:
            # コメントが生成されなかった、または失敗した場合
            st.warning(f"'{table_name}': AIコメントが有効に生成されなかったため、ベクトルは生成されません。")

        # 3. メタデータテーブルに保存
        # スピナー内のテキストを更新 (ログで代替)
        logger.info(f"{database_name}.{schema_name}.{table_name}: メタデータを保存中...")
        # 保存するデータを辞書にまとめる
        update_data = {
            "TABLE_COMMENT": generated_comment, # AIが生成したコメント (失敗文字列の場合もある)
            "ANALYSIS_IDEAS": json.dumps(ideas, ensure_ascii=False), # アイデアリストをJSON文字列に変換して保存
            "EMBEDDING": embedding # 生成されたベクトル (Noneの場合もある)
            # LIKES はこの関数では更新しない (いいねボタンで更新)
            # SOURCE_TABLE_COMMENT は情報スキーマから取得するものなので、ここでは保存しない
        }

        # update_metadata 関数を呼び出して保存処理を実行
        # spinner は with ブロック終了時に自動で消える
        save_successful = update_metadata(database_name, schema_name, table_name, update_data)

    # 保存処理の結果に基づいてメッセージを表示
    if save_successful:
        st.success(f"'{table_name}' のメタデータ (コメント, アイデア, ベクトル) を生成・保存しました。")
        # キャッシュクリア (get_all_metadata などが最新情報を反映するように)
        st.cache_data.clear()
        return True # 成功
    else:
        # update_metadata 内でエラーが表示されているはずだが、念のため
        st.error(f"'{table_name}' のメタデータ保存に失敗しました。")
        return False # 失敗


# --- データリネージ関連関数 ---
@st.cache_data(ttl=1800) # 結果を30分キャッシュ
def get_dynamic_lineage(target_database, target_schema, target_table, direction='upstream', max_depth=3, time_window_days=90):
    """
    指定されたテーブルを起点として、`snowflake.account_usage.access_history` から
    動的なデータネージ(データの流れ、依存関係) を取得します。
    現在は上流方向 (upstream: どのテーブルからデータが来たか) のみサポートします。
    Args:
        target_database (str): 起点テーブルのデータベース名。
        target_schema (str): 起点テーブルのスキーマ名。
        target_table (str): 起点テーブル名。
        direction (str): リネージを辿る方向 ('upstream' のみ現在サポート)。
        max_depth (int): 遡る最大のステップ数 (深さ)。
        time_window_days (int): 検索対象とする履歴の期間 (日数)。
    Returns:
        dict or None: {'nodes': list, 'edges': list} の形式の辞書。
                      ノードは {'id': 完全修飾名, 'label': 表示名, 'domain': タイプ}。
                      エッジは {'source': ..., 'target': ..., 'query_id': ...}。
                      エラー発生時は None。
    """
    # 現在は upstream のみサポート
    if direction != 'upstream':
        st.warning("現在、動的リネージは上流方向（upstream）のみサポートしています。")
        return None

    # リネージ取得開始のログ
    logger.info(f"動的リネージ取得開始: {target_database}.{target_schema}.{target_table}, max_depth={max_depth}, days={time_window_days}")
    # 結果を格納するセット (ノード: (完全修飾名, ドメイン), エッジ: (ソースID, ターゲットID, クエリID))
    nodes = set() # 重複を避けるためセットを使用
    edges = set() # 重複を避けるためセットを使用
    # 起点となるテーブルの完全修飾名 (大文字)
    start_node_id = f"{target_database}.{target_schema}.{target_table}".upper()
    # 探索対象のノードを管理するリスト (タプル: (ノードID, 現在の深さ))
    nodes_to_process = [(start_node_id, 0)]
    # 処理済みのノードを記録するセット (無限ループ防止用)
    processed_nodes = set()

    # --- 必要な Account Usage ビューへのアクセス権限チェック ---
    required_views = ["snowflake.account_usage.access_history", "snowflake.account_usage.query_history"]
    for view_name in required_views:
        try:
            # 各ビューに対して簡単なクエリを実行してアクセス可能か確認
            session.sql(f"SELECT 1 FROM {view_name} LIMIT 1").collect()
        except SnowparkSQLException as check_err:
            # 権限エラーまたは存在しないエラー
            if "does not exist or not authorized" in str(check_err):
                logger.error(f"{view_name} へのアクセス権限エラー: {check_err}")
                st.error(f"データリネージ取得に必要な `{view_name}` へのアクセス権限がありません。")
                return None # 権限がなければリネージ取得不可
            else:
                # その他のSQLエラー
                logger.error(f"{view_name} のチェック中にエラー: {check_err}")
                st.error(f"データリネージ情報の取得中にエラーが発生しました: {check_err}")
                return None
        except Exception as e:
             # 予期せぬエラー
             logger.error(f"{view_name} チェック中に予期せぬエラー: {e}", exc_info=True)
             st.error(f"データリネージ情報の取得中に予期せぬエラーが発生しました: {e}")
             return None

    # --- 起点ノードの情報を取得 ---
    try:
        # INFORMATION_SCHEMA から起点テーブルのタイプ (TABLE, VIEW など) を取得
        domain_query = f"SELECT TABLE_TYPE FROM {target_database}.INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? LIMIT 1;"
        params = [target_schema, target_table] # パラメータでスキーマ名とテーブル名を指定
        domain_res = session.sql(domain_query, params=params).collect()
        # 結果があれば TABLE_TYPE を取得、なければデフォルトで 'TABLE' とする
        start_node_domain = domain_res[0]['TABLE_TYPE'] if domain_res else 'TABLE'
        # 起点ノードをノードセットに追加
        nodes.add((start_node_id, start_node_domain))
    except Exception as e:
        # ドメイン取得に失敗した場合 (テーブルが存在しないなど)
        logger.warning(f"起点ノードのドメイン取得失敗 ({start_node_id}): {e}")
        # ドメイン不明 ('UNKNOWN' またはデフォルト 'TABLE') としてノードを追加
        nodes.add((start_node_id, 'TABLE')) # または 'UNKNOWN'

    # --- リネージ探索の開始 ---
    # 検索対象とする期間の開始日時を計算 (UTC)
    start_time_limit = (datetime.utcnow() - timedelta(days=time_window_days)).strftime('%Y-%m-%d %H:%M:%S')

    # 探索対象リスト (nodes_to_process) が空になるまでループ
    while nodes_to_process:
        # リストの先頭から探索対象ノードと現在の深さを取得
        current_node_id, current_depth = nodes_to_process.pop(0)

        # 最大深度に達したか、既に処理済みのノードであればスキップ
        if current_depth >= max_depth or current_node_id in processed_nodes:
            continue

        # このノードを処理済みとしてマーク
        processed_nodes.add(current_node_id)
        # デバッグログ
        logger.debug(f"深度 {current_depth} で探索中: {current_node_id}")

        try:
            # --- ACCESS_HISTORY を検索するクエリ ---
            # 目的: `current_node_id` にデータを書き込んだクエリ (`query_id`) を特定し、
            #       そのクエリが読み取り元としたオブジェクト (`source_object_name`) を見つける。
            # ステップ:
            # 1. `ModifiedObjects` CTE: `objects_modified` 配列をフラット化し、
            #    `current_node_id` が含まれるレコードの `query_id` を取得。
            #    `query_start_time` で期間を絞り込む。NULLチェックも追加。
            # 2. `AccessedSources` CTE:
            #    - `ModifiedObjects` で特定した `query_id` について、
            #      `direct_objects_accessed` (直接アクセス) 配列をフラット化し、
            #      ソースオブジェクト名 (`source_object_name`) とドメイン (`source_object_domain`) を取得。
            #      自分自身 (`current_node_id`) へのアクセスは除外。NULLの場合エラーにならないようCOALESCE。
            #    - 同様に `base_objects_accessed` (基盤アクセス) についてもソースを取得。
            #      StageやQuery自体はリネージの対象外とする。
            #    - `UNION` で direct と base の結果を結合 (重複排除)。
            # 3. 最終結果: `AccessedSources` から `query_id`, `source_object_name`, `source_object_domain` を選択。
            #    `source_object_name` が NULL のものは除外。
            query = f"""
            WITH ModifiedObjects AS (
                -- 1. 対象オブジェクトに書き込んだクエリを特定
                SELECT DISTINCT ah.query_id
                FROM snowflake.account_usage.access_history ah
                -- objects_modified 配列を展開
                CROSS JOIN TABLE(FLATTEN(input => ah.objects_modified)) mod_obj
                WHERE ah.query_start_time >= '{start_time_limit}' -- 期間絞り込み
                  -- value内の objectName フィールドが現在のノードIDと一致するか (文字列比較)
                  AND mod_obj.value:"objectName"::string = '{current_node_id}'
                  -- objectId が NULL でないことを確認 (より確実な同定のため推奨)
                  AND mod_obj.value:"objectId" IS NOT NULL
            ), AccessedSources AS (
                -- 2. 上記クエリがアクセスしたオブジェクト（ソース）を特定
                SELECT
                    mo.query_id,
                    -- direct_objects_accessed 配列を展開し、objectName と objectDomain を取得
                    acc.value:"objectName"::string as source_object_name,
                    acc.value:"objectDomain"::string as source_object_domain
                FROM snowflake.account_usage.access_history ah
                JOIN ModifiedObjects mo ON ah.query_id = mo.query_id -- 書き込みクエリで絞り込み
                -- direct_objects_accessed が NULL の場合もエラーにならないよう空配列に変換
                CROSS JOIN TABLE(FLATTEN(input => COALESCE(ah.direct_objects_accessed, PARSE_JSON('[]')))) acc
                WHERE acc.value:"objectName"::string IS NOT NULL -- ソース名がNULLでない
                  AND acc.value:"objectName"::string != '{current_node_id}' -- 自分自身は除外

                UNION -- direct と base の結果を結合 (重複は自動で排除される)

                SELECT
                    mo.query_id,
                    -- base_objects_accessed 配列を展開し、objectName と objectDomain を取得
                    base_acc.value:"objectName"::string as source_object_name,
                    base_acc.value:"objectDomain"::string as source_object_domain
                FROM snowflake.account_usage.access_history ah
                JOIN ModifiedObjects mo ON ah.query_id = mo.query_id
                -- base_objects_accessed が NULL の場合もエラーにならないよう空配列に変換
                CROSS JOIN TABLE(FLATTEN(input => COALESCE(ah.base_objects_accessed, PARSE_JSON('[]')))) base_acc
                WHERE base_acc.value:"objectName"::string IS NOT NULL -- ソース名がNULLでない
                  AND base_acc.value:"objectName"::string != '{current_node_id}' -- 自分自身は除外
                  -- Stage や Query 自体はリネージのノードとして扱わない
                  AND base_acc.value:"objectDomain"::string NOT IN ('Stage', 'Query')
            )
            -- 3. 結果を選択 (AccessedSources から重複排除済み)
            SELECT query_id, source_object_name, source_object_domain
            FROM AccessedSources
            WHERE source_object_name IS NOT NULL; -- ソース名が NULL のレコードは除外
            """

            # クエリを実行し、結果を Pandas DataFrame に変換
            results_df = session.sql(query).to_pandas()
            # デバッグログ: 見つかったソースの数
            logger.debug(f"{current_node_id} への書き込み元クエリ結果: {len(results_df)} 件")

            # --- クエリ結果を処理してノードとエッジを追加 ---
            for _, row in results_df.iterrows():
                # 結果行からソースオブジェクト名、ドメイン、クエリIDを取得
                source_id_full_val = row.get('SOURCE_OBJECT_NAME')
                source_domain_val = row.get('SOURCE_OBJECT_DOMAIN')
                query_id = row.get('QUERY_ID')

                # ソースオブジェクト名またはクエリIDが取得できなかった場合はスキップ
                if not source_id_full_val or not query_id:
                    logger.warning(f"ソースオブジェクト名またはQuery IDがNULLのためスキップ: {row.to_dict()}")
                    continue

                # ソースID (完全修飾名) を大文字に変換
                source_id_full = str(source_id_full_val).upper()
                # ソースドメインを取得 (NULLなら 'UNKNOWN')、大文字に変換
                source_domain = str(source_domain_val).upper() if source_domain_val else 'UNKNOWN'

                # ソースIDが有効な形式か簡易チェック (例: DB.SCHEMA.TABLE のように '.' が2つ以上あるか)
                # これにより、不完全な名前や予期しない形式のオブジェクトを除外
                if len(source_id_full.split('.')) < 3 and source_domain not in ('STAGE'): # STAGEはDB.SCHEMAを含まないことがある
                    logger.warning(f"無効なソースオブジェクト名をスキップ: {source_id_full} (from query {query_id})")
                    continue

                # --- ノードとエッジを追加 ---
                # 新しいソースノードをノードセットに追加 (重複は自動で無視される)
                nodes.add((source_id_full, source_domain))
                # エッジ情報をタプルとして作成 (ソースID, ターゲットID, クエリID)
                edge = (source_id_full, current_node_id, query_id)

                # このエッジがまだ追加されていなければ追加
                if edge not in edges:
                    edges.add(edge)
                    # 新しいエッジ発見のログ
                    logger.debug(f"新しい動的エッジ発見: {source_id_full} -> {current_node_id} (Query: {query_id})")
                    # 次の探索対象として、このソースノードをリストに追加
                    # ただし、最大深度に達していない、まだ処理されていない、探索リストにまだない場合のみ
                    if current_depth + 1 < max_depth and \
                       source_id_full not in processed_nodes and \
                       not any(n[0] == source_id_full for n in nodes_to_process):
                        nodes_to_process.append((source_id_full, current_depth + 1))

        except SnowparkSQLException as e:
             # リネージ取得クエリの実行中にSQLエラーが発生した場合
             logger.error(f"動的リネージクエリ中にSQLエラー ({current_node_id}): {e}", exc_info=True)
             # UIに警告を表示 (処理は続行される可能性がある)
             st.warning(f"リネージ情報の一部取得中にSQLエラーが発生しました: {e}")
        except Exception as e:
             # その他の予期せぬエラー
             logger.error(f"動的リネージ取得中に予期せぬエラー ({current_node_id}): {e}", exc_info=True)
             st.warning(f"リネージ情報の一部取得中に予期せぬエラーが発生しました: {e}")

    # --- ループ完了後、結果を整形して返す ---
    # ノードセットを辞書のリスト形式に変換
    # id: 完全修飾名, label: 表示用の短い名前 (通常はオブジェクト名), domain: タイプ
    result_nodes = [{'id': n[0], 'label': n[0].split('.')[-1], 'domain': n[1]} for n in nodes]
    # エッジセットを辞書のリスト形式に変換
    # source: ソースID, target: ターゲットID, query_id: 関連クエリID
    result_edges = [{'source': e[0], 'target': e[1], 'query_id': e[2]} for e in edges]

    # 最終的なノード数とエッジ数をログに出力
    logger.info(f"動的リネージ取得完了: ノード数={len(result_nodes)}, エッジ数={len(result_edges)}")
    # 整形した結果の辞書を返す
    return {'nodes': result_nodes, 'edges': result_edges}


# データリネージグラフを Graphviz を使って描画する関数
def create_lineage_graph(nodes, edges, start_node_id):
    """
    ノードとエッジのリストから Graphviz の Digraph オブジェクトを生成します。
    ノードの形状をタイプ (domain) に応じて変え、起点ノードを強調表示します。
    エッジにはツールチップでクエリIDなどの情報を表示します。
    Args:
        nodes (list): ノード情報の辞書のリスト [{'id': ..., 'label': ..., 'domain': ...}]。
        edges (list): エッジ情報の辞書のリスト [{'source': ..., 'target': ..., 'query_id': ...}]。
        start_node_id (str): 強調表示する起点ノードの完全修飾名 (大文字)。
    Returns:
        graphviz.Digraph: 生成されたGraphvizグラフオブジェクト。
    """
    # Graphviz Digraph オブジェクトを作成 (有向グラフ)
    # comment: グラフのコメント
    # graph_attr: グラフ全体の属性設定 ('rankdir': 'LR' は左から右へのレイアウト)
    dot = graphviz.Digraph(comment='Data Lineage', graph_attr={'rankdir': 'LR'})

    # ノードのドメイン (タイプ) と Graphviz での形状のマッピング
    shape_map = {
        'TABLE': 'box',              # テーブルは四角
        'VIEW': 'ellipse',           # ビューは楕円
        'MATERIALIZED VIEW': 'box3d', # マテビューは3Dボックス
        'STREAM': 'cds',             # ストリームは円筒 (Cylinder/Database Shape)
        'TASK': 'component',         # タスクはコンポーネント形状
        'PIPE': 'cylinder',          # パイプは円筒
        'FUNCTION': 'septagon',      # 関数は七角形
        'PROCEDURE': 'octagon',      # プロシージャは八角形
        'STAGE': 'folder',           # ステージはフォルダ形状
        'EXTERNAL TABLE': 'tab',     # 外部テーブルはタブ形状
        'UNKNOWN': 'question',       # 不明なタイプは疑問符
        # 必要に応じて他のタイプも追加
    }

    # --- ノードの描画 ---
    processed_node_ids = set() # 重複してノードを描画しないようにIDを記録
    for node in nodes:
        node_id = node['id'] # ノードの完全修飾名
        # 既に描画済みのノードIDはスキップ
        if node_id in processed_node_ids: continue
        processed_node_ids.add(node_id) # 描画済みとして記録

        node_label = node['label'] # 表示ラベル (テーブル名など)
        node_domain = node.get('domain', 'UNKNOWN').upper() # ノードタイプ (大文字), 不明なら 'UNKNOWN'
        # shape_map から対応する形状を取得、なければデフォルト 'ellipse'
        shape = shape_map.get(node_domain, 'ellipse')
        # ノードにマウスオーバーした際に表示されるツールチップテキスト
        node_tooltip = f"テーブルタイプ: {node_domain}\n完全修飾テーブル名: {node_id}"

        # 起点ノードかどうかを判定してスタイルを設定
        if node_id == start_node_id.upper():
            # 起点ノード: 強調表示 (薄い赤色で塗りつぶし)
            dot.node(node_id, label=node_label, shape=shape, style='filled', fillcolor='lightcoral', tooltip=node_tooltip)
        else:
            # それ以外のノード: 標準表示 (薄い青色で塗りつぶし)
            dot.node(node_id, label=node_label, shape=shape, style='filled', fillcolor='lightblue', tooltip=node_tooltip)

    # --- エッジの描画 ---
    processed_edges = set() # 重複してエッジを描画しないように (source, target) タプルを記録
    for edge in edges:
        # ソースとターゲットが同じエッジ (自己参照) は描画しない
        if edge['source'] == edge['target']:
            continue

        # エッジのタプル (source, target) を作成
        edge_tuple = (edge['source'], edge['target'])
        # 既に描画済みのエッジはスキップ
        if edge_tuple in processed_edges: continue
        processed_edges.add(edge_tuple) # 描画済みとして記録

        # エッジに関連付けられたクエリIDを取得 (存在すれば)
        query_id = edge.get('query_id')
        # エッジにマウスオーバーした際に表示されるツールチップテキスト
        edge_tooltip = f"Source: {edge['source']}\nTarget: {edge['target']}"
        if query_id:
            # クエリIDがあればツールチップに追加
            edge_tooltip += f"\nQuery ID: {query_id}"
        # エッジを描画 (ラベルは省略し、ツールチップで情報提供)
        dot.edge(edge['source'], edge['target'], tooltip=edge_tooltip)

    # 生成された Graphviz オブジェクトを返す
    return dot

# テーブル情報をカード形式で表示する関数
def display_table_card(table_info, metadata):
    """
    一つのテーブルに関する情報 (基本情報、AIメタデータ、アクセス数、リネージなど) を
    Streamlit のコンテナ (カード形式) で表示します。
    Args:
        table_info (dict): テーブルの基本情報を含む辞書。最低限 'DATABASE_NAME',
                           'SCHEMA_NAME', 'TABLE_NAME' が必要。'TABLE_TYPE',
                           'SOURCE_TABLE_COMMENT', 'ROW_COUNT', 'BYTES', 'CREATED',
                           'LAST_ALTERED', 'search_similarity' などが含まれる想定。
        metadata (dict): AIによって生成されたメタデータやいいね数などを含む辞書。
                         `get_metadata` または `get_all_metadata` の結果。
    """
    # --- 必須情報の取得とチェック ---
    db_name = table_info.get('DATABASE_NAME')
    sc_name = table_info.get('SCHEMA_NAME')
    tbl_name = table_info.get('TABLE_NAME')

    # 必須情報が欠けている場合はエラーログを出力して終了
    if not db_name or not sc_name or not tbl_name:
        logger.error(f"display_table_card に必須情報 (DB, Schema, Table名) が不足しています: {table_info}")
        return

    # テーブルを一意に識別するキー (完全修飾名)
    table_key = f"{db_name}.{sc_name}.{tbl_name}"
    # Streamlit ウィジェットのキーを生成 (特殊文字をアンダースコアに置換して衝突を避ける)
    elem_key_base = re.sub(r'\W+', '_', table_key)

    # メタデータが辞書でない場合 (予期せぬ入力)、警告ログを出して空辞書で代替
    if not isinstance(metadata, dict):
        logger.warning(f"display_table_cardに不正な型のmetadataが渡されました: {type(metadata)} for key {table_key}")
        metadata = {}

    # --- カード表示用のコンテナを作成 (枠線付き) ---
    card = st.container(border=True)

    # --- カードヘッダー: テーブル名(短縮)といいねボタン --- 
    col1, col2 = card.columns([0.85, 0.15])
    with col1:
        # テーブル名のみを太字で表示 
        col1.markdown(f"**{tbl_name}**")
    with col2:
        # 現在のいいね数を取得 (メタデータになければデフォルト0)
        current_likes = metadata.get("LIKES", 0)
        # いいねボタンを表示 (ユニークなキーを設定)
        if col2.button(f"👍 {current_likes}", key=f"like_{elem_key_base}", help="いいね！"):
            # ボタンが押されたら LIKES_INCREMENT を True にしてメタデータ更新
            if update_metadata(db_name, sc_name, tbl_name, {"LIKES_INCREMENT": True}):
                 # 成功したらトーストメッセージを表示
                 st.toast(f"「{tbl_name}」にいいねしました！", icon="👍")
                 # キャッシュをクリアして表示を更新するために再実行
                 st.cache_data.clear()
                 st.rerun()
            else:
                 # 失敗したらエラーメッセージを表示
                 st.error("いいねの更新に失敗しました。")

    # --- 検索類似度 (キーワード検索時にあれば表示) ---
    # table_info 辞書から 'search_similarity' キーの値を取得
    search_similarity = table_info.get('search_similarity')
    # 値が存在し、かつ NaN でない場合のみ表示
    if search_similarity is not None and pd.notna(search_similarity):
         # キャプションとして類似度をパーセント表示
         card.caption(f"検索キーワードとの関連度: {search_similarity:.2%}")

    # --- LLMによるテーブル概要  --- 
    # メタデータからLLM生成コメントを取得 (なければ None)
    llm_comment = metadata.get("TABLE_COMMENT", None)
    # コメントが存在し、かつ「生成失敗」文字列を含まない場合
    if llm_comment and "生成失敗" not in str(llm_comment):
        # コメントをキャプションで表示
        card.caption(llm_comment)
    # コメントが「生成失敗」を含む場合
    elif "生成失敗" in str(llm_comment):
         # 失敗メッセージとエラー詳細を表示
         card.caption(f"AIによる概要生成に失敗しました。({llm_comment})")
    # コメントが存在しない場合 (未生成)
    else:
        card.caption("未生成")
        # メタデータ未生成の場合、LLM生成ボタンを表示
        # 元のテーブルコメント (source_table_comment) を取得試行 (プロンプト用)
        source_comment_orig = table_info.get('SOURCE_TABLE_COMMENT')
        # LLM生成ボタン (ユニークなキーを設定)
        if card.button("LLMで概要と分析アイデアを生成", key=f"gen_ai_{elem_key_base}"):
             # AIメタデータ生成・保存関数を呼び出し (デフォルトモデルを使用)
             generate_and_save_ai_metadata(db_name, sc_name, tbl_name, source_comment_orig) # model引数はデフォルト
             # 処理完了後、表示を更新するために再実行
             st.rerun()

    # --- 詳細表示 (Expander) ---
    # Expander を使って詳細情報を折りたたみ表示
    with card.expander("詳細を表示"):

        # --- データベース名とスキーマ名 --- 
        st.markdown("**データベース / スキーマ情報:**")
        st.text(f"データベース名: {db_name}")
        st.text(f"スキーマ名: {sc_name}")
        st.divider() # 区切り線

        # --- LLMによる分析アイデア ---
        st.markdown("**LLMによる分析アイデア/ユースケース:**")
        # メタデータからLLM生成分析アイデア (JSON文字列) を取得 (なければ "[]")
        analysis_ideas_str = metadata.get("ANALYSIS_IDEAS", "[]")
        analysis_ideas = [] # 解析結果のリスト (初期値は空)
        # JSON文字列が存在し、かつ文字列型の場合のみ解析
        if analysis_ideas_str and isinstance(analysis_ideas_str, str):
            try:
                # JSON文字列をPythonリストにパース
                ideas_parsed = json.loads(analysis_ideas_str)
                # パース結果がリストであれば採用
                if isinstance(ideas_parsed, list):
                    analysis_ideas = ideas_parsed
                else:
                     # リストでない場合は不正形式として扱う
                     analysis_ideas = ["(分析アイデアの形式が不正)"]
            except (json.JSONDecodeError, TypeError):
                # JSONデコード失敗、または予期せぬ型の場合
                logger.warning(f"ANALYSIS_IDEASのJSONデコード失敗: {table_key}")
                analysis_ideas = ["(分析アイデアの形式が不正)"]

        # アイデアリストが存在し、かつ「生成失敗」文字列を含む要素がない場合
        if analysis_ideas and not any("生成失敗" in str(idea) for idea in analysis_ideas):
            # 各アイデアを箇条書きで表示
            for idea in analysis_ideas:
                st.caption(f"- {idea}")
        # アイデアリストに「生成失敗」が含まれる場合
        elif any("生成失敗" in str(idea) for idea in analysis_ideas):
             # 失敗メッセージと内容を表示
             st.caption(f"LLMによる分析アイデア生成に失敗しました。({analysis_ideas})")
        # コメントは生成済みだがアイデアがない場合 (通常は同時に生成されるはず)
        elif llm_comment and "生成失敗" not in str(llm_comment):
             st.caption("未生成")
        # コメントもアイデアも未生成の場合
        else:
            st.caption("未生成 (概要と同時に生成されます)")

        st.divider() # 区切り線

        # --- 元のテーブルコメント (INFORMATION_SCHEMA) ---
        # table_info から元のコメントを取得
        source_comment_display = table_info.get('SOURCE_TABLE_COMMENT')
        # コメントが存在し、かつ NaN でない場合
        if source_comment_display and pd.notna(source_comment_display):
            st.markdown("**元のテーブルコメント (Information Schema):**")
            # キャプションで表示
            st.caption(source_comment_display)
            st.divider() # 区切り線

        # --- アクセス数とテーブル情報 ---
        # カラムを使ってアクセス数とその他の情報を横に並べる
        col_meta1, col_meta2 = st.columns(2)

        with col_meta1:
             st.markdown("**直近1ヶ月のアクセス数:**")
             # アクセス数取得関数を呼び出し
             monthly_access = get_monthly_access_count(db_name, sc_name, tbl_name)
             # st.metric で見やすく表示
             st.metric(label="アクセス回数 (クエリ単位)", value=monthly_access)

        with col_meta2:
            st.markdown("**テーブル情報:**")
            # 表示したい情報を table_info と metadata から集める
            info_dict = {
                "タイプ": table_info.get('TABLE_TYPE', 'N/A'),   # テーブルタイプ
                "行数": table_info.get('ROW_COUNT', 'N/A'),    # 行数
                "サイズ(Bytes)": table_info.get('BYTES', 'N/A'), # サイズ
                "作成日時": table_info.get('CREATED'),       # 作成日時 (information_schema)
                "最終更新日時": table_info.get('LAST_ALTERED'), # 最終更新日時 (information_schema)
                "メタデータ最終更新": metadata.get('LAST_REFRESHED') # AIメタデータ更新日時
            }
            # 表示用に整形するための空辞書
            display_info = {}
            # info_dict の各項目を処理
            for k, v in info_dict.items():
                 # 値が存在し、'N/A' でなく、NaN でない場合のみ処理
                 if v is not None and v != 'N/A' and pd.notna(v):
                    # --- 値の型に応じて表示形式を調整 ---
                    # 整数または浮動小数点数の場合
                    if isinstance(v, (int, float)):
                        if k == "行数":
                            # 3桁区切りで表示 (負の値はN/A扱い)
                            display_info[k] = f"{v:,.0f}" if v >= 0 else "N/A"
                        elif k == "サイズ(Bytes)":
                            # バイト数を GB/MB/KB/Bytes に変換して表示 (負の値はN/A扱い)
                            if v >= 1024**3: display_info[k] = f"{v / 1024**3:.2f} GB"
                            elif v >= 1024**2: display_info[k] = f"{v / 1024**2:.2f} MB"
                            elif v >= 1024: display_info[k] = f"{v / 1024:.2f} KB"
                            elif v >= 0: display_info[k] = f"{v} Bytes"
                            else: display_info[k] = "N/A"
                        else:
                            # その他の数値はそのまま
                            display_info[k] = v
                    # 日時型 (datetime または pandas Timestamp) の場合
                    elif isinstance(v, (datetime, pd.Timestamp)):
                        try:
                            # タイムゾーン情報があれば考慮してフォーマット、なければそのままフォーマット
                            if hasattr(v, 'tzinfo') and v.tzinfo: # タイムゾーン情報の有無を確認
                                display_info[k] = v.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')
                            else:
                                display_info[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as dt_err: # より具体的なエラーハンドリング
                            logger.warning(f"Datetime formatting error for key '{k}', value '{v}': {dt_err}")
                            display_info[k] = str(v) # エラー時は元の文字列表現
                    # その他の型は文字列に変換
                    else:
                         display_info[k] = str(v)

            # 整形された表示情報がある場合
            if display_info:
                # キーと値をテキストで表示
                for key, val in display_info.items():
                    st.text(f"{key}: {val}")
            else:
                # 表示する情報がない場合
                st.caption("追加情報なし")

         # --- データリネージ (動的依存関係) ---
        st.divider() # 区切り線
        st.markdown("**データリネージ（データの流れ - 上流）**")
        # リネージ取得のパラメータ設定用ウィジェット
        # 遡る日数の入力 (数値入力、デフォルト90日、最小1日、最大365日)
        lineage_days = st.number_input("遡る日数", min_value=1, max_value=365, value=90, step=1, key=f"lineage_days_{elem_key_base}", help="この日数前までのデータ操作履歴（ACCESS_HISTORY）を検索します。長くすると時間がかかります。")
        # 遡る深さの入力 (数値入力、デフォルト3、最小1、最大5)
        lineage_depth = st.number_input("遡る深さ", min_value=1, max_value=5, value=3, step=1, key=f"lineage_depth_{elem_key_base}", help="何ステップ前までデータの流れを遡るか指定します。深くすると時間がかかります。")

        # リネージ機能に関する説明キャプション
        st.caption(f"`ACCESS_HISTORY`ビューに基づき、過去{lineage_days}日間のデータの流れ（どのテーブル/ビューからデータが来たか）を表示します。最大{3}時間程度のデータ遅延が発生する場合があります。検索範囲が広いと時間がかかることがあります。")

        # --- リネージグラフの表示/非表示制御 ---
        # 表示状態を管理するためのセッションステートキー
        lineage_visible_key = f"lineage_visible_{elem_key_base}"
        # ボタンを配置するためのプレースホルダー
        lineage_button_placeholder = st.empty()

        # セッションステートでリネージが非表示状態の場合
        if not st.session_state.get(lineage_visible_key, False):
            # 「表示/更新」ボタンを表示
            if lineage_button_placeholder.button("リネージを表示/更新", key=f"lineage_btn_show_{elem_key_base}"):
                # ボタンが押されたら、表示状態を True に設定
                st.session_state[lineage_visible_key] = True
                # 状態変更を反映するために再実行
                st.rerun()

        # セッションステートでリネージが表示状態の場合
        if st.session_state.get(lineage_visible_key, False):
            # 「非表示」ボタンを表示
            if lineage_button_placeholder.button("リネージを非表示", key=f"lineage_btn_hide_{elem_key_base}"):
                # ボタンが押されたら、表示状態を False に設定
                st.session_state[lineage_visible_key] = False
                # 状態変更を反映するために再実行
                st.rerun()

            # リネージグラフを表示するためのプレースホルダー
            lineage_placeholder = st.empty()
            # プレースホルダー内にグラフを描画
            with lineage_placeholder.container():
                # グラフ描画中のメッセージを表示
                st.info(f"過去{lineage_days}日間のリネージ情報を最大{lineage_depth}ステップ遡って取得・描画中...")
                try:
                    # 動的リネージ取得関数を呼び出し (日数と深さをパラメータで渡す)
                    lineage_data = get_dynamic_lineage(db_name, sc_name, tbl_name, direction='upstream', max_depth=lineage_depth, time_window_days=lineage_days)

                    # --- リネージ取得結果に応じた処理 ---
                    # 結果が None の場合 (権限エラーなどで取得失敗)
                    if lineage_data is None:
                         # get_dynamic_lineage 内でエラーメッセージ表示済みのため、ここでは警告のみ
                         st.warning("リネージ情報の取得に失敗しました。権限や設定を確認してください。")
                    # ノードが存在し、かつ起点ノード以外にもノードがある場合 (リネージが見つかった)
                    elif lineage_data['nodes'] and len(lineage_data['nodes']) > 1:
                        # 起点ノードの完全修飾名 (大文字)
                        start_node_full_id = f"{db_name}.{sc_name}.{tbl_name}".upper()
                        # グラフ生成関数を呼び出し
                        graph = create_lineage_graph(lineage_data['nodes'], lineage_data['edges'], start_node_full_id)
                        # Streamlit で Graphviz グラフを表示
                        st.graphviz_chart(graph)
                    # ノードは存在するが、起点ノードのみの場合 (上流が見つからなかった)
                    elif lineage_data['nodes']:
                         st.info(f"過去{lineage_days}日間の履歴では、このオブジェクトの上流データは見つかりませんでした。")
                    # データ取得には成功したが、ノードが空の場合 (通常は起こらないはず)
                    else:
                        st.warning("リネージ情報の取得に失敗しました（データが空）。")

                # --- Graphviz 関連のエラーハンドリング ---
                except ImportError:
                     # graphviz ライブラリがインポートできない場合
                     logger.error("Graphvizライブラリが見つかりません。")
                     st.error("リネージ表示に必要なGraphvizライブラリがインストールされていない可能性があります。")
                except graphviz.backend.execute.ExecutableNotFound:
                     # graphviz の実行ファイル (dotコマンドなど) が見つからない場合
                     logger.error("Graphviz実行ファイルが見つかりません。")
                     st.error("Graphviz実行ファイルが見つかりません。環境にインストールし、パスを通してください。")
                # その他の予期せぬエラー
                except Exception as e:
                     logger.error(f"リネージ表示エラー ({table_key}): {e}", exc_info=True)
                     st.error(f"リネージ表示中に予期せぬエラー: {e}")

# データカタログのメインページを表示する関数
def main_page():
    """データカタログのメインページ（検索、フィルター、結果表示）を構成・表示します。"""
    # ページヘッダー
    st.header("データカタログ")

    # --- メタデータテーブルの存在確認・作成 ---
    # セッションステートでテーブル作成済みか確認 (初回実行時のみチェック)
    if 'metadata_table_created' not in st.session_state:
        # テーブル作成関数を呼び出し
        if create_metadata_table():
            # 成功したらセッションステートに記録
            st.session_state.metadata_table_created = True
        else:
            # 失敗したらエラーメッセージを表示して処理中断
            st.error("メタデータテーブルの準備に失敗したため、処理を中断します。")
            return # メインページの処理をこれ以上進めない

    # --- 状態管理用のコールバック関数 ---
    def reset_search_state():
        """DB/スキーマ選択が変更されたときに、検索状態をリセットするコールバック"""
        st.session_state['current_view'] = 'browse' # 表示モードを「閲覧」に
        st.session_state['search_results_data'] = None # 検索結果データをクリア
        st.session_state['last_search_term'] = ""    # 最後の検索語をクリア
        # 必要であれば、検索入力欄のキー 'search_input' の値もクリア
        # st.session_state['search_input'] = ""
        logger.debug("Search state reset due to DB/Schema selection change.")

    # --- サイドバー (検索 & フィルター) ---
    st.sidebar.header("1. 検索 & フィルター")
    # キーワード検索入力欄 (キー 'search_input')
    search_term = st.sidebar.text_input("キーワード検索 (全テーブル対象)", key="search_input")
    # ベクトル検索有効化トグル (キー 'search_vector_toggle')
    search_vector = st.sidebar.toggle("ベクトル検索を有効にする", value=True, help=f"検索語とLLM生成コメントのベクトル類似度({DEFAULT_EMBEDDING_MODEL}, {EMBEDDING_DIMENSION}次元)で検索します。", key="search_vector_toggle")
    # ベクトル検索時の類似度閾値 (デフォルト 0.7)
    similarity_threshold = 0.7
    # ベクトル検索が有効な場合のみスライダーを表示 (キー 'similarity_slider')
    if search_vector:
        similarity_threshold = st.sidebar.slider("類似度の閾値", 0.0, 1.0, 0.7, 0.05, key="similarity_slider")
    # 検索実行ボタン (キー 'search_button')
    search_button = st.sidebar.button("検索実行", key="search_button")

    st.sidebar.divider() # 区切り線

    st.sidebar.header("2. 特定のテーブルを検索")
    # データベース選択ドロップダウン (キー 'db_select')
    # on_change に検索状態リセット用コールバックを設定
    db_options = get_databases()
    selected_db = st.sidebar.selectbox("データベースで絞り込み", options=db_options, index=0, key="db_select", on_change=reset_search_state)
    # スキーマ選択マルチセレクト (キー 'schema_select')
    # on_change に検索状態リセット用コールバックを設定
    selected_schemas = []
    if selected_db != SELECT_OPTION: # データベースが選択されている場合のみ表示
        schema_options = get_schemas_for_database(selected_db)
        if schema_options:
             selected_schemas = st.sidebar.multiselect("スキーマで絞り込み", options=schema_options, default=[], key="schema_select", on_change=reset_search_state)
             # 注意: multiselect の変更でも on_change が呼ばれる

    # --- メイン表示エリア ---
    # 状態表示用プレースホルダー (検索中メッセージなど)
    status_placeholder = st.empty()
    # 結果表示用コンテナ
    results_container = st.container()

    # --- セッションステートの初期化 (初回実行時) ---
    if 'current_view' not in st.session_state:
        st.session_state['current_view'] = 'browse' # 初期表示は 'browse' モード
    if 'search_results_data' not in st.session_state:
        st.session_state['search_results_data'] = None # 検索結果データ (初期値 None)
    if 'last_search_term' not in st.session_state:
        st.session_state['last_search_term'] = "" # 最後に実行された検索語

    # --- 検索ボタンが押された場合の処理 ---
    if search_button and search_term:
        # 表示モードを 'search_results' に設定
        st.session_state['current_view'] = 'search_results'
        # 検索語を保存
        st.session_state['last_search_term'] = search_term
        # 検索中メッセージを表示
        status_placeholder.info("全テーブルを対象に検索を実行中...")
        # 検索語を小文字に変換 (キーワード検索用)
        search_lower = search_term.lower()
        # 最終的な検索結果を格納する DataFrame
        final_results_df = pd.DataFrame()
        # ベクトル検索を試みたかどうかのフラグ
        vector_search_executed = False
        # ベクトル検索SQLが成功したかどうかのフラグ
        vector_search_succeeded = False
        # ベクトル検索時のパラメータ (初期値)
        vector_params = []
        # キーワード検索時のパラメータ (初期値)
        keyword_params = []
        # 実行されたSQL (エラーログ用)
        executed_sql = ""

        try:
            # --- クエリで使用するカラムリスト ---
            # メタデータテーブルから取得する基本カラム
            select_columns_metadata = """
                database_name, schema_name, table_name, table_comment,
                analysis_ideas, likes, last_refreshed
            """ # embeddingはベクトル検索時のみ直接使う

            # --- ベクトル検索が有効な場合の処理 ---
            if search_vector:
                vector_search_executed = True # 試行フラグON
                status_placeholder.text("検索語のベクトルを計算中...") # メッセージ更新
                # 検索語 (パラメータで渡すのでエスケープ不要)
                escaped_search_term = search_term

                # --- ベクトル検索SQL (CTEを使用) ---
                # 1. search_vector CTE: 検索語のベクトルを計算
                # 2. メインクエリ: メタデータテーブルと CTE を結合し、
                #    保存されている embedding と検索語ベクトルのコサイン類似度を計算 (SIMILARITY)
                #    embedding が NULL でないレコードのみ対象
                #    類似度で降順ソート (NULLS LAST で類似度がないものを最後に)
                vector_search_sql = f"""
                WITH search_vector AS (
                    SELECT {EMBED_FUNCTION_NAME}(?, ?) as query_embedding
                )
                SELECT
                    m.{select_columns_metadata.replace(',', ', m.')}, -- エイリアス m を追加
                    VECTOR_COSINE_SIMILARITY(m.embedding, sv.query_embedding) as SIMILARITY
                FROM
                    {METADATA_TABLE_NAME} m, search_vector sv -- テーブルエイリアスを使用
                WHERE
                    m.embedding IS NOT NULL -- ベクトルが存在するもののみ
                ORDER BY
                    similarity DESC NULLS LAST -- 類似度でソート
                """
                executed_sql = vector_search_sql # ログ用に保存
                # パラメータ: 使用する Embedding モデル名、検索語
                vector_params = [DEFAULT_EMBEDDING_MODEL, escaped_search_term]

                logger.info(f"Executing vector search query (using CTE)") # SQL自体はログに出さない方が安全な場合も
                logger.debug(f"Vector search params: {[DEFAULT_EMBEDDING_MODEL, '<search_term>']}") # 検索語はマスク

                # ベクトル検索SQLを実行し、結果を DataFrame に取得
                vector_results_df = session.sql(vector_search_sql, params=vector_params).to_pandas()
                vector_search_succeeded = True # SQL実行自体は成功
                logger.info(f"Vector search query returned {len(vector_results_df)} results.")
                logger.debug(f"Vector search results columns: {vector_results_df.columns.tolist()}")

                # --- Python側でのフィルタリング (類似度閾値) ---
                status_placeholder.text("検索結果をフィルタリング中...") # メッセージ更新
                filtered_rows = [] # フィルタ後の行を格納するリスト
                # 結果が存在し、'SIMILARITY' カラムがある場合
                if not vector_results_df.empty and 'SIMILARITY' in vector_results_df.columns:
                    for index, row in vector_results_df.iterrows():
                        # 類似度を取得 (NaNの場合は0とする)
                        similarity = row.get('SIMILARITY', 0.0)
                        similarity = similarity if pd.notna(similarity) else 0.0

                        # 類似度が閾値以上の場合
                        if similarity >= similarity_threshold:
                            row_dict = row.to_dict() # 行を辞書に変換
                            # 表示用に 'search_similarity' キーを追加
                            row_dict['search_similarity'] = similarity
                            filtered_rows.append(row_dict) # リストに追加

                    # フィルタ後の辞書リストから DataFrame を再作成
                    final_results_df = pd.DataFrame(filtered_rows)
                    # 必要なら再度ソート (SQLでソート済みのはずだが念のため)
                    # if not final_results_df.empty:
                    #     final_results_df = final_results_df.sort_values(by='search_similarity', ascending=False)
                else:
                     logger.warning("Vector search results missing 'SIMILARITY' column or empty.")
                     final_results_df = pd.DataFrame() # 結果なし

            # --- ベクトル検索が無効な場合の処理 (キーワード検索のみ) ---
            else:
                # キーワード検索 SQL: DB名, スキーマ名, テーブル名, AIコメント, AIアイデア に LIKE 検索
                # 類似度カラム (SIMILARITY) は NULL で追加し、列構造をベクトル検索と合わせる
                keyword_search_sql = f"""
                SELECT
                    {select_columns_metadata}
                    , NULL as SIMILARITY -- 類似度列をNULLで追加
                FROM {METADATA_TABLE_NAME}
                WHERE
                    (
                        LOWER(database_name) LIKE ? OR LOWER(schema_name) LIKE ? OR
                        LOWER(table_name) LIKE ? OR LOWER(table_comment) LIKE ? OR
                        LOWER(analysis_ideas) LIKE ?
                    )
                ORDER BY database_name, schema_name, table_name -- 名前順でソート
                """
                executed_sql = keyword_search_sql # ログ用に保存
                # LIKE検索用のパラメータ (%検索語%)
                keyword_param = f"%{search_lower}%"
                # 5つのLIKE条件に対応するパラメータリスト
                keyword_params = [keyword_param] * 5

                logger.info(f"Executing keyword search query") # SQL自体はログ非推奨
                logger.debug(f"Keyword search params: {keyword_params}") # パラメータもマスク推奨
                # キーワード検索SQLを実行し、結果を DataFrame に取得
                final_results_df = session.sql(keyword_search_sql, params=keyword_params).to_pandas()
                logger.info(f"Keyword search query returned {len(final_results_df)} results.")
                # 表示用の類似度列 'search_similarity' を追加し、None を設定
                final_results_df['search_similarity'] = None

        # --- エラーハンドリング (SQL実行およびPython処理) ---
        except SnowparkSQLException as e:
            status_placeholder.error(f"検索クエリの実行中にSQLエラーが発生しました: {e}")
            logger.error(f"Search query execution failed: {e}", exc_info=True)
            # 失敗したSQLとパラメータをログに出力 (デバッグ用、機密情報に注意)
            failed_sql = executed_sql
            failed_params = vector_params if vector_search_executed else keyword_params
            logger.error(f"Failed SQL (summary): {'Vector search' if vector_search_executed else 'Keyword search'}")
            # logger.error(f"Failed SQL: {failed_sql}") # SQL全体をログに出すか検討
            logger.error(f"Failed Params: {failed_params}") # パラメータはマスク推奨
            final_results_df = pd.DataFrame() # エラー時は空のDataFrame
        except Exception as e:
            status_placeholder.error(f"検索処理中に予期せぬエラーが発生しました: {e}")
            logger.error(f"Unexpected error during search: {e}", exc_info=True)
            final_results_df = pd.DataFrame() # エラー時は空のDataFrame

        # --- 検索結果をセッションステートに保存 ---
        # DataFrame を辞書のリスト形式に変換して保存 (セッションステートに適した形式)
        if not final_results_df.empty:
            # final_results_df にはメタデータ情報のみ含まれるため、
            # 表示に必要な情報 (TABLE_TYPE, SOURCE_TABLE_COMMENT など) を追加する必要がある。
            # -> information_schema から取得した情報をマージするか、
            #    表示時に都度 get_tables_for_database_schema を呼ぶ必要がある。
            #    ここでは、シンプルにメタデータテーブル由来の情報のみをstateに保存し、
            #    表示時に get_all_metadata と get_tables_for_database_schema の情報を組み合わせて表示する方針とする。
            #    (検索結果表示のロジックを修正)
            #    -> いや、表示時に information_schema 情報を再度引くのは非効率。
            #       検索結果 (final_results_df) に information_schema 情報をマージしてから state に保存する。

            # --- information_schema 情報を取得してマージ ---
            status_placeholder.text("テーブル基本情報を取得中...")
            all_tables_info = {} # DBごとにテーブル情報をキャッシュ
            merged_results = []
            processed_dbs = set() # information_schema 取得済みDBを記録
            try:
                for idx, row_meta in final_results_df.iterrows():
                    db_s = row_meta['DATABASE_NAME']
                    sc_s = row_meta['SCHEMA_NAME']
                    tbl_s = row_meta['TABLE_NAME']
                    if db_s not in processed_dbs:
                        # DB単位で information_schema 情報を取得 (キャッシュ利用)
                        # 検索結果に含まれるスキーマのみを対象にすることも可能だが、ここではDB全体を取得
                        logger.debug(f"Fetching table info for DB: {db_s}")
                        info_df = get_tables_for_database_schema(db_s, None) # 全スキーマ取得
                        if not info_df.empty:
                            # DataFrameを行ごとに辞書にして、キー(DB.SCHEMA.TABLE)で引けるようにする
                            for _, info_row in info_df.iterrows():
                                key = f"{info_row['DATABASE_NAME']}.{info_row['SCHEMA_NAME']}.{info_row['TABLE_NAME']}"
                                all_tables_info[key] = {k:v for k,v in info_row.items() if pd.notna(v)}
                        processed_dbs.add(db_s)

                    # マージ処理
                    table_key_s = f"{db_s}.{sc_s}.{tbl_s}"
                    base_info = all_tables_info.get(table_key_s, {}) # information_schema情報
                    meta_info = {k:v for k,v in row_meta.items() if pd.notna(v)} # メタデータ情報

                    # 両方の情報をマージ (メタデータ情報を優先する場合)
                    merged_info = {**base_info, **meta_info}
                    # search_similarity はメタデータ由来なので確実に保持
                    if 'search_similarity' in meta_info:
                        merged_info['search_similarity'] = meta_info['search_similarity']

                    # 必須キーが揃っているか再確認
                    if 'DATABASE_NAME' in merged_info and 'SCHEMA_NAME' in merged_info and 'TABLE_NAME' in merged_info:
                        merged_results.append(merged_info)
                    else:
                         logger.warning(f"Failed to merge info for {table_key_s}. Meta: {meta_info}, Base: {base_info}")

                # マージされた結果 (辞書のリスト) をセッションステートに保存
                st.session_state['search_results_data'] = merged_results
                logger.info(f"Merged search results with table info: {len(merged_results)} items.")
            except Exception as merge_err:
                status_placeholder.error("テーブル基本情報の取得・マージ中にエラーが発生しました。")
                logger.error(f"Error during merging table info: {merge_err}", exc_info=True)
                # マージ失敗時はメタデータのみの結果を保存（表示がおかしくなる可能性あり）
                st.session_state['search_results_data'] = final_results_df.to_dict('records')

        else:
            # 検索結果が0件の場合
            st.session_state['search_results_data'] = [] # 空リストを保存

        # 検索中メッセージを消去
        status_placeholder.empty()
        # この時点で st.rerun() は不要。下の表示ロジックで session_state が参照される。

    # --- 結果表示ロジック ---
    with results_container:
        # === 検索結果表示モード ('search_results') ===
        if st.session_state.get('current_view') == 'search_results':
            # 保存された検索語を使ってサブヘッダーを表示
            st.subheader(f"検索結果: '{st.session_state.get('last_search_term', '')}'")
            # セッションステートから検索結果データ (マージ済み辞書のリスト) を取得
            search_results_list = st.session_state.get('search_results_data', [])

            # 検索結果の件数などの情報を表示
            search_info = f"{len(search_results_list)} 件のテーブルが見つかりました。"
            st.info(search_info)

            # 検索結果リストが存在する場合
            if search_results_list:
                # 全メタデータを取得 (いいね数などを表示するため)
                all_metadata_display = get_all_metadata()
                # 結果を3カラムで表示
                cols = st.columns(3)
                col_idx = 0
                # 保存された結果リスト (辞書のリスト) をループ
                for table_info_search_dict in search_results_list:
                    # 各テーブルの情報辞書を取得 (マージ済み)
                    db_name_search = table_info_search_dict.get('DATABASE_NAME')
                    sc_name_search = table_info_search_dict.get('SCHEMA_NAME')
                    tbl_name_search = table_info_search_dict.get('TABLE_NAME')

                    # 必須キーがなければスキップ
                    if not db_name_search or not sc_name_search or not tbl_name_search:
                        logger.warning(f"Search result dict is missing required keys after merge: {table_info_search_dict}")
                        continue

                    # メタデータ辞書を取得 (all_metadata_display から)
                    table_key_search = f"{db_name_search}.{sc_name_search}.{tbl_name_search}"
                    metadata_search = all_metadata_display.get(table_key_search, {}) # いいね数などを反映

                    # 対応するカラムにテーブルカードを表示
                    with cols[col_idx % 3]:
                        # display_table_card には マージ済み情報辞書 と メタデータ辞書を渡す
                        display_table_card(table_info_search_dict, metadata_search)
                    col_idx += 1
            # 検索結果が空リストの場合
            elif st.session_state.get('last_search_term'): # 検索語がある場合のみメッセージ表示
                 st.warning("指定されたキーワード/類似度に一致するテーブルは見つかりませんでした。")

        # === DB/スキーマ選択による閲覧モード ('browse') ===
        elif st.session_state.get('current_view') == 'browse':
            # データベースが選択されている場合
            if selected_db != SELECT_OPTION:
                # 選択されたDB/スキーマのテーブルを取得
                status_placeholder.info(f"データベース '{selected_db}' のテーブルを取得中...")
                # selected_schemas は on_change で更新された最新の値が使われる
                schemas_to_fetch_browse = tuple(selected_schemas) if selected_schemas else None
                tables_df_browse = get_tables_for_database_schema(selected_db, schemas_to_fetch_browse)
                status_placeholder.empty() # 取得完了したらメッセージ削除

                # サブヘッダー表示
                schema_disp = f' (スキーマ: {", ".join(selected_schemas)})' if selected_schemas else ''
                st.subheader(f"テーブル一覧: {selected_db}{schema_disp}")
                st.info(f"{len(tables_df_browse)} 件のテーブル/ビューが見つかりました。")

                # テーブル一覧が存在する場合
                if not tables_df_browse.empty:
                    # 全メタデータを取得 (いいね数などを表示するため)
                    all_metadata_browse = get_all_metadata()
                    # 3カラムで表示
                    cols_browse = st.columns(3)
                    col_idx_browse = 0
                    # 取得したテーブル情報の DataFrame をループ
                    for index, table_row_browse in tables_df_browse.iterrows():
                        try:
                            # DataFrameの行を辞書に変換 (NaNを除外)
                            table_info_browse = {k: v for k, v in table_row_browse.items() if pd.notna(v)}
                            # 必須キー (DB, Schema, Table) が存在するか確認
                            db_b = table_info_browse.get('DATABASE_NAME')
                            sc_b = table_info_browse.get('SCHEMA_NAME')
                            tbl_b = table_info_browse.get('TABLE_NAME')
                            if not db_b or not sc_b or not tbl_b:
                                logger.warning(f"Browse result row (index {index}) is missing required keys.")
                                continue
                            # メタデータ辞書を取得
                            table_key_browse = f"{db_b}.{sc_b}.{tbl_b}"
                            metadata_browse = all_metadata_browse.get(table_key_browse, {})
                            # 対応するカラムにテーブルカードを表示
                            with cols_browse[col_idx_browse % 3]:
                                display_table_card(table_info_browse, metadata_browse)
                            col_idx_browse += 1
                        except Exception as e:
                            # カード表示中のエラーハンドリング
                            logger.error(f"Error processing browse result row (index {index}): {e}", exc_info=True)
                            st.error(f"テーブル {table_row_browse.get('TABLE_NAME', index)} の表示中にエラーが発生しました。")
                # テーブルが見つからなかった場合
                else:
                    st.warning("指定された条件に一致するテーブル/ビューはありません。")
            # データベースも選択されていない場合
            else:
                status_placeholder.info("サイドバーで検索を実行するか、データベースを選択してテーブルを表示してください。")

        # === その他の状態 (通常は発生しないはず) ===
        else:
             status_placeholder.info("サイドバーで検索を実行するか、データベースを選択してテーブルを表示してください。")


# --- 管理ページ ---
def admin_page():
    """管理機能（メタデータ一括生成など）のページを構成・表示します。"""
    # ページヘッダー
    st.header("管理機能")

    # --- メタデータテーブルの存在確認 ---
    # セッションステートで確認 (メインページで作成されているはずだが念のため)
    if 'metadata_table_created' not in st.session_state:
        # 作成されていなければ試行
        if create_metadata_table():
            st.session_state.metadata_table_created = True
        else:
            # 作成失敗なら管理機能は利用不可
            st.error("メタデータテーブルの準備に失敗したため、管理機能を利用できません。")
            return

    # --- メタデータ一括生成セクション ---
    st.subheader("LLMによるメタデータ生成")
    st.markdown(f"""
    - 選択したデータベース・スキーマ配下のテーブルについて、以下のメタデータをLLMで一括生成し、メタデータテーブル (`{METADATA_TABLE_NAME}`) に保存します。
        - **LLMによるテーブルコメント** (`TABLE_COMMENT`)
        - **分析アイデア** (`ANALYSIS_IDEAS`, JSON文字列)
        - **ベクトル表現** (`EMBEDDING`, テーブルコメントから生成)

    **設定:**
    *   **使用するLLMモデル** を以下から選択してください。モデルによって性能やコストが異なります。
    *   対象とする **データベース** と **スキーマ** を選択してください (スキーマ未選択時はDB内の全スキーマが対象)。

    **注意:**
    *   既にメタデータが存在する場合、**新しい情報で上書きされます** (いいね数は保持されます)。
    *   テーブル数が多い場合、処理に時間がかかり、Snowflakeのクレジットを消費します。
    *   `SNOWFLAKE.CORTEX.COMPLETE` および `{EMBED_FUNCTION_NAME}` 関数へのアクセス権限が必要です。
    *   ベクトル生成に使用するモデルは `{DEFAULT_EMBEDDING_MODEL}` (次元数: {EMBEDDING_DIMENSION}) で固定です。
    """)

    # デフォルトモデルがリストに含まれているか確認し、インデックスを取得
    try:
        default_model_index = AVAILABLE_LLM_MODELS.index(DEFAULT_LLM_MODEL)
    except ValueError:
        default_model_index = 0 # デフォルトが見つからない場合は最初のモデルを選択
        st.warning(f"デフォルトLLMモデル '{DEFAULT_LLM_MODEL}' が選択肢リストにありません。リストの最初のモデルをデフォルトにします。")

    selected_llm_model = st.selectbox(
        "コメント/アイデア生成に使用するLLMモデルを選択",
        options=AVAILABLE_LLM_MODELS,
        index=default_model_index, # デフォルトモデルを選択
        key="admin_llm_select",
        help="テーブルコメントと分析アイデアの生成に使用するモデルです。"
    )

    # --- 対象選択 ---
    # カラムでDB選択とスキーマ選択を横に並べる
    col1, col2 = st.columns(2)
    with col1:
        # データベース選択ドロップダウン (管理ページ用)
        db_options_admin = get_databases()
        # '<Select>' オプションを除外
        db_options_admin = [db for db in db_options_admin if db != SELECT_OPTION]
        selected_db_admin = st.selectbox(
            "対象データベースを選択",
            options=db_options_admin,
            index=0 if db_options_admin else -1, # リストが空でなければ最初のDBを選択
            key="admin_db_select" # ユニークなキー
        )

    with col2:
        # スキーマ選択マルチセレクト (管理ページ用)
        schema_options_admin = []
        selected_schemas_admin = []
        if selected_db_admin: # DBが選択されている場合のみ
            schema_options_admin = get_schemas_for_database(selected_db_admin)
            if schema_options_admin:
                 selected_schemas_admin = st.multiselect(
                     "対象スキーマを選択 (未選択時は全スキーマ)",
                     options=schema_options_admin,
                     default=[], # デフォルトは空リスト (全スキーマ選択)
                     key="admin_schema_select" # ユニークなキー
                 )
            else:
                # スキーマ取得失敗時のメッセージ
                st.caption(f"'{selected_db_admin}' 内のスキーマを取得できませんでした。")
        else:
             # DB未選択時のメッセージ
             st.caption("データベースを選択してください。")

    # --- 生成対象プレビューと実行ボタン ---
    if selected_db_admin: # DBが選択されている場合のみ表示
        st.markdown("---") # 区切り線
        st.markdown("**生成対象テーブルのプレビュー**")

        try:
            # 選択されたDB/スキーマのテーブル一覧を取得 (キャッシュ利用)
            schemas_to_fetch_admin = tuple(selected_schemas_admin) if selected_schemas_admin else None
            target_tables_df = get_tables_for_database_schema(selected_db_admin, schemas_to_fetch_admin)

            # 対象テーブルが存在する場合
            if not target_tables_df.empty:
                # --- メタデータの存在状況をプレビューに追加 ---
                # 全メタデータを取得 (キャッシュ利用)
                all_meta_admin = get_all_metadata()
                # 各テーブルのメタデータキー (DB.SCHEMA.TABLE) を作成
                target_tables_df['metadata_key'] = target_tables_df.apply(
                    lambda row: f"{row['DATABASE_NAME']}.{row['SCHEMA_NAME']}.{row['TABLE_NAME']}",
                    axis=1
                )
                # メタデータレコード自体が存在するか (キーが all_meta_admin にあるか)
                target_tables_df['metadata_exists'] = target_tables_df['metadata_key'].isin(all_meta_admin)
                # LLM生成コメントが存在するか (None や失敗文字列でないか)
                target_tables_df['llm_comment_exists'] = target_tables_df['metadata_key'].apply(
                    lambda key: all_meta_admin.get(key, {}).get('TABLE_COMMENT') not in [None, "AIコメント生成失敗", "AIコメント生成失敗(応答抽出エラー)", "AIコメント生成失敗(不正な形式)", "AIコメント生成失敗(JSONパースエラー)", "AIコメント生成失敗(処理エラー)"] # エラーパターンを追加
                )
                # Embeddingが存在するか (Noneでないか)
                target_tables_df['embedding_exists'] = target_tables_df['metadata_key'].apply(
                    lambda key: all_meta_admin.get(key, {}).get('EMBEDDING') is not None
                )

                # プレビュー表示するカラムを選択
                display_cols = ['DATABASE_NAME', 'SCHEMA_NAME', 'TABLE_NAME', 'TABLE_TYPE', 'llm_comment_exists', 'embedding_exists']
                # DataFrame を表示 (高さ制限付き)
                st.dataframe(target_tables_df[display_cols], use_container_width=True, height=300)

                # 対象テーブル数と上書き件数を計算
                total_target_count = len(target_tables_df)
                overwrite_count = target_tables_df['metadata_exists'].sum()
                # 上書きがある場合のみ注記を追加
                overwrite_notice = f" (うち {overwrite_count} 件は既存メタデータを上書き)" if overwrite_count > 0 else ""

                # 対象件数と上書き情報を表示
                st.info(f"{total_target_count} 件のテーブルが対象です{overwrite_notice}。")

                # --- 一括生成実行ボタン ---
                # ボタンのラベルに対象件数を含める (ユニークなキーを設定)
                button_label = f"{total_target_count} 件のテーブルのメタデータを生成/更新する (モデル: {selected_llm_model})"
                if st.button(button_label, type="primary", key="admin_generate_button"):

                    # --- 一括生成処理ループ ---
                    # ループ全体の進捗バー
                    admin_loop_progress_bar = st.progress(0)
                    # 各テーブル処理中のステータス表示用プレースホルダー
                    admin_loop_status_placeholder = st.empty()
                    # 成功/失敗カウンター
                    success_count = 0
                    fail_count = 0

                    # ボタンが押された後に再度テーブル一覧を取得（キャッシュ利用のはずだが念のため）
                    # このDataFrameをループ処理で使用
                    process_tables_df = get_tables_for_database_schema(selected_db_admin, schemas_to_fetch_admin)
                    total_process_count = len(process_tables_df) # 処理対象件数を再計算

                    # DataFrame の各行 (テーブル) に対して処理を実行
                    for i, (_, table_row) in enumerate(process_tables_df.iterrows()):
                        # テーブル情報を取得
                        db = table_row['DATABASE_NAME']
                        sc = table_row['SCHEMA_NAME']
                        tbl = table_row['TABLE_NAME']
                        # 元のテーブルコメントを取得 (AIプロンプト用)
                        src_comment = table_row.get('SOURCE_TABLE_COMMENT')

                        # 現在処理中のテーブル名をステータス表示
                        admin_loop_status_placeholder.text(f"全体進捗 ({i+1}/{total_process_count}): {db}.{sc}.{tbl} を処理中 (モデル: {selected_llm_model})...")

                        try:
                            # --- 個別テーブルのメタデータ生成・保存関数を呼び出し ---
                            if generate_and_save_ai_metadata(db, sc, tbl, src_comment, model=selected_llm_model):
                                # 成功したらカウンターを増やす
                                success_count += 1
                            else:
                                # 失敗したらカウンターを増やす (エラーメッセージは関数内で表示済み)
                                fail_count += 1
                        except Exception as e:
                             # generate_and_save_ai_metadata 呼び出し自体で予期せぬエラーが発生した場合
                             fail_count += 1
                             logger.error(f"メタデータ生成ループ内で予期せぬエラー ({db}.{sc}.{tbl}, model={selected_llm_model}): {e}", exc_info=True)
                             # ステータス表示エリアにエラー情報を一時的に表示
                             admin_loop_status_placeholder.error(f"テーブル {tbl} の処理中に予期せぬエラー発生。ログを確認してください。", icon="🔥")
                             # エラーが発生してもループは継続し、次のテーブルの処理に進む

                        # ループ全体のプログレスバーを更新
                        admin_loop_progress_bar.progress((i + 1) / total_process_count)

                    # --- ループ完了後 ---
                    # ループ用のプログレスバーとステータス表示をクリア
                    admin_loop_progress_bar.empty()
                    admin_loop_status_placeholder.empty()

                    # 最終結果メッセージを表示
                    if success_count > 0:
                        st.success(f"{success_count} 件のテーブルのメタデータ生成/更新が完了しました (モデル: {selected_llm_model})。")
                    if fail_count > 0:
                        st.warning(f"{fail_count} 件のテーブルの処理に失敗またはスキップされました。詳細はアプリログを確認してください。")
                    # 処理対象が0件だった場合のメッセージは不要 (ボタン押下前に件数表示されるため)

                    # キャッシュをクリアして表示を最新化
                    st.cache_data.clear()
                    st.rerun() # ページを再読み込みして結果を反映

            # 対象テーブルが見つからなかった場合
            else:
                st.warning("指定されたデータベース/スキーマにテーブルが見つかりませんでした。")

        # テーブル取得/プレビュー中のエラーハンドリング
        except Exception as e:
            st.error(f"対象テーブルの取得またはプレビュー表示中にエラーが発生しました: {e}")
            logger.error(f"管理ページでのテーブル取得/プレビューエラー: {e}", exc_info=True)

    # DBが選択されていない場合のメッセージ
    else:
        st.info("メタデータを生成するデータベースを選択してください。")


# アプリケーションのエントリーポイント
def main():
    """アプリケーション全体の制御、ページ選択を行います。"""
    # --- ページ選択ラジオボタン (サイドバー) ---
    page = st.sidebar.radio(
        "ページ選択",
        ["データカタログ", "管理"], # 選択肢
        key="page_selection" # ウィジェットのキー
    )

    # --- 選択されたページに応じて対応する関数を実行 ---
    if page == "データカタログ":
        # データカタログページを表示
        main_page()
    elif page == "管理":
        # 管理ページを表示
        admin_page()

# --- アプリケーションの実行 ---
if __name__ == "__main__":
    main()