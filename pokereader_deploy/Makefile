run_streamlit:
	streamlit run app.py

run_st:
	streamlit run app.py

run_api:
	uvicorn api.main:app --reload --port=8200

docker_image:
	docker build -t ${DOCKER_NAME}:${DOCKER_VERSION} .

docker_image_nc:
	docker build --no-cache -t ${DOCKER_NAME}:${DOCKER_VERSION} .

docker_push:
	docker push ${DOCKER_NAME}:${DOCKER_VERSION}

docker_deploy_local:
	docker run -e PORT=8080 -p 8080:8080 ${DOCKER_NAME}:${DOCKER_VERSION}

docker_deploy_gcp:
	gcloud run deploy --image ${DOCKER_NAME}:${DOCKER_VERSION} --region ${GCP_REGION} --memory=1Gi

