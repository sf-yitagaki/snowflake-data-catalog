
-- ロールの指定
USE ROLE accountadmin;
USE WAREHOUSE compute_wh;

CREATE OR REPLACE DATABASE data_catalog;
USE SCHEMA public;


-- 全てのモデルにアクセスするためにクロスリージョンコールの設定
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- ステージの作成
CREATE OR REPLACE STAGE data_catalog.public.catalog_stage encryption = (type = 'snowflake_sse') DIRECTORY = (ENABLE = TRUE);

-- Git連携のため、API統合を作成する
CREATE OR REPLACE API INTEGRATION git_api_integration
  API_PROVIDER = git_https_api
  API_ALLOWED_PREFIXES = ('https://github.com/sf-yitagaki/')
  ENABLED = TRUE;

-- GIT統合の作成
CREATE OR REPLACE GIT REPOSITORY GIT_INTEGRATION_FOR_CATALOG
  API_INTEGRATION = git_api_integration
  ORIGIN = 'https://github.com/sf-yitagaki/snowflake-data-catalog.git';

-- Githubからファイルを持ってくる
COPY FILES INTO @data_catalog.public.catalog_stage FROM @GIT_INTEGRATION_FOR_CATALOG/branches/main/environment.yml;

-- Streamlit in Snowflakeの作成
CREATE OR REPLACE STREAMLIT data_catalog
    FROM @GIT_INTEGRATION_FOR_CATALOG/branches/main
    MAIN_FILE = 'sis.py'
    QUERY_WAREHOUSE = COMPUTE_WH
    ENVIRONMENT_FILE = 'environment.yml';