-- Run once as the postgres superuser. Then point DATABASE_URL at app_ro.
CREATE ROLE app_ro LOGIN PASSWORD 'password123';  -- 1. Creates a new database login role named 'app_ro' with a password     
GRANT CONNECT ON DATABASE moviedb TO app_ro;    -- 2. Allows 'app_ro' to connect to the 'moviedb' database    
GRANT USAGE ON SCHEMA public TO app_ro;         -- 3. Allows 'app_ro' to access the 'public' schema 
GRANT SELECT ON movies, tag_lookup TO app_ro;   -- 4. Grants READ-ONLY (SELECT) access strictly to the movies and tag_lookup tables
ALTER ROLE app_ro SET statement_timeout = '5s'; -- 5. Automatically aborts/kills any SQL query run by this user that takes longer than 5 seconds 
