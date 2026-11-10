-- Schema for the PostgreSQL face store.
--
-- The library creates this table on its own when it connects (unless
-- Options::create_schema is turned off), so applying this file by hand is
-- only needed when the runtime user is not allowed to create tables.
--
-- One row per enrolled face sample. The embedding is the 128-element
-- single-precision vector SFace produces; the photograph it was computed
-- from is never stored.
--
--   psql "$DATABASE_URL" -f sql/schema.sql

CREATE TABLE IF NOT EXISTS face_enrollments (
    id          bigserial   PRIMARY KEY,
    person_id   integer     NOT NULL,
    embedding   real[]      NOT NULL,
    enrolled_at timestamptz NOT NULL DEFAULT now()
);

-- Enrolling a person and removing one both look the rows up by person id.
CREATE INDEX IF NOT EXISTS face_enrollments_person_id_idx
    ON face_enrollments (person_id);

-- A person may be enrolled more than once (several samples of the same
-- face). The application picks the closest sample, so no uniqueness
-- constraint is wanted here.
