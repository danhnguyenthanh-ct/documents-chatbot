start-qdrant:
	docker run -d --name qdrant-server -p 6333:6333 qdrant/qdrant:latest

stop-qdrant:
	docker stop qdrant-server
	docker rm qdrant-server
