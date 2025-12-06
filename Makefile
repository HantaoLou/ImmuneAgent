build-ui:
	cd ui  && npm install&& npm run build

deploy-ui: build-ui
	cp -r ui/dist /opt/antibody_gen/

run-server:
	cd agent && uv run main.py

run: deploy-ui
	cd agent && uv run main.py

fmt:
	cd agent && uvx ruff check --select I --fix && uvx ruff format && cd ../ui && npm run format 