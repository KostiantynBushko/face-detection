-- Schema for the PostgreSQL face store.
--
-- The library creates this table on its own when it connects (unless
-- Options::create_schema is turned off), so applying this file by hand is
-- only needed when the runtime user is not allowed to create tables.
--
-- One row per enrolled face sample. The embedding is the 128-element
-- single-precision vector SFace produces.
--
-- The three optional columns hold the properties of the person a sample
-- belongs to:
--
--   user_name -- the name that is shown on screen when the person is
--                detected
--   photo     -- a picture of the face, as the bytes of an encoded image
--   metadata  -- an arbitrary JSON document
--
-- They were added after the table was introduced, so a table that predates
-- them has to be migrated. The ALTER statements below do that; they are
-- harmless on a table that already has the columns.
--
--   psql "$DATABASE_URL" -f sql/schema.sql

CREATE TABLE IF NOT EXISTS face_enrollments (
    id          bigserial   PRIMARY KEY,
    person_id   integer     NOT NULL,
    embedding   real[]      NOT NULL,
    user_name   text,
    photo       bytea,
    metadata    jsonb       NOT NULL DEFAULT '{}'::jsonb,
    enrolled_at timestamptz NOT NULL DEFAULT now()
);

-- An installation that enrolled faces before the properties were added
-- still has the old table. The columns are added here; the ones that are
-- already there are left alone.
ALTER TABLE face_enrollments ADD COLUMN IF NOT EXISTS user_name text;
ALTER TABLE face_enrollments ADD COLUMN IF NOT EXISTS photo bytea;
ALTER TABLE face_enrollments ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Enrolling a person and removing one both look the rows up by person id.
CREATE INDEX IF NOT EXISTS face_enrollments_person_id_idx
    ON face_enrollments (person_id);

-- A person may be enrolled more than once (several samples of the same
-- face). The application picks the closest sample, so no uniqueness
-- constraint is wanted here.
