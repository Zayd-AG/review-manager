-- Adds nullable fields needed by the backend models.
-- Existing reviews and clusters remain unchanged.

ALTER TABLE reviews
    ADD COLUMN IF NOT EXISTS category VARCHAR(64),
    ADD COLUMN IF NOT EXISTS severity VARCHAR(16),
    ADD COLUMN IF NOT EXISTS justification TEXT;

ALTER TABLE clusters
    ADD COLUMN IF NOT EXISTS category VARCHAR(64),
    ADD COLUMN IF NOT EXISTS severity VARCHAR(16);
