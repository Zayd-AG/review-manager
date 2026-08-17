-- Adds nullable fields needed by the backend models.
-- Existing reviews and clusters remain unchanged.

ALTER TABLE reviews
    ADD COLUMN IF NOT EXISTS category VARCHAR(64),
    ADD COLUMN IF NOT EXISTS severity VARCHAR(16),
    ADD COLUMN IF NOT EXISTS justification TEXT;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'clusters'
    ) THEN
        ALTER TABLE clusters
            ADD COLUMN IF NOT EXISTS category VARCHAR(64),
            ADD COLUMN IF NOT EXISTS severity VARCHAR(16);
    END IF;
END $$;
