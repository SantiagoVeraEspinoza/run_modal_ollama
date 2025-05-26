import asyncio
import subprocess
import modal
import os

MODEL_DIR = "/ollama_models"
ORIGINAL_MODEL = "qwen3:14b"
CUSTOM_MODEL = "qwen3-14b"
OLLAMA_VERSION = "0.6.5"
OLLAMA_PORT = 11434

ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "ca-certificates")
    .pip_install(
        "fastapi==0.115.8",
        "uvicorn[standard]==0.34.0",
        "openai~=1.30",
    )
    .run_commands(
        "echo 'Installing Ollama...'",
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
        f"mkdir -p {MODEL_DIR}",
    )
    .env(
        {
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,
        }
    )
)

app = modal.App("ollama-server", image=ollama_image)

model_volume = modal.Volume.from_name("ollama-models-store", create_if_missing=True)

@app.cls(
    gpu="T4",
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 5,
    min_containers=1,
)
class OllamaServer:
    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        print("Starting Ollama server...")
        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        await asyncio.sleep(10)  # Let Ollama boot

        loop = asyncio.get_running_loop()
        models_updated = False

        # Always remove the existing custom model if it exists
        print(f"Deleting old model: {CUSTOM_MODEL} (if exists)")
        delete_proc = await asyncio.create_subprocess_exec("ollama", "rm", CUSTOM_MODEL)
        await delete_proc.wait()
        
        # # Always remove the existing original model if it exists - COMMENT IF FASTER MODEL DEPLOY IS NEEDED
        # print(f"Deleting original model: {ORIGINAL_MODEL} (if exists)")
        # delete_proc = await asyncio.create_subprocess_exec("ollama", "rm", ORIGINAL_MODEL)
        # await delete_proc.wait()

        # Always pull the base model if not present
        ollama_list_proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True
        )
        current_models_output = ollama_list_proc.stdout

        if ORIGINAL_MODEL not in current_models_output:
            print(f"Pulling model: {ORIGINAL_MODEL}")
            pull_proc = await asyncio.create_subprocess_exec("ollama", "pull", ORIGINAL_MODEL)
            await pull_proc.wait()
            models_updated = True

        # Create fresh Modelfile
        print(f"Creating Modelfile for {CUSTOM_MODEL}")
        modelfile_path = f"/tmp/Modelfile.{CUSTOM_MODEL}"
        with open(modelfile_path, "w") as f:
            # f.write(f"FROM {ORIGINAL_MODEL}\nPARAMETER num_ctx 16384\n")
            f.write(f"FROM {ORIGINAL_MODEL}\n")

        # Build the new custom model
        print(f"Building custom model: {CUSTOM_MODEL}")
        create_proc = await asyncio.create_subprocess_exec(
            "ollama", "create", CUSTOM_MODEL, "-f", modelfile_path
        )
        await create_proc.wait()
        models_updated = True

        # Save changes to Modal volume
        if models_updated:
            print("Committing updated model volume...")
            await loop.run_in_executor(None, model_volume.commit)

    @modal.exit()
    def stop_ollama(self):
        if self.ollama_process and self.ollama_process.poll() is None:
            try:
                self.ollama_process.terminate()
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                self.ollama_process.wait()

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=180)
    def serve(self):
        print(f"Ollama server exposed on port {OLLAMA_PORT}")
