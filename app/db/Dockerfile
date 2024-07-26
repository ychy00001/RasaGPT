FROM ankane/pgvector:v0.5.1

# Copy in the load-extensions script
COPY ./create_db.sh /docker-entrypoint-initdb.d/
RUN chmod +x /docker-entrypoint-initdb.d/create_db.sh