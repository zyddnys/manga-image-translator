build-image:
	docker rmi manga-image-translator || true
	docker build . --tag=manga-image-translator

run-web-server:
	docker run --gpus all -p 5003:5003 --ipc=host --rm manga-image-translator python /app/translate_demo.py \
		--target-lang=ENG \ 
		--manga2eng \ 
		--verbose \
		--log-web \
		--mode web \
		--use-inpainting \
		--use-cuda \ 
		--host=0.0.0.0 \
		--port=5003