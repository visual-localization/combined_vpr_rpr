import modal
import shlex
import subprocess
from pathlib import Path

STREAMLIT_DIR = Path(__file__).parent / "streamlit_app"
REMOTE_STREAMLIT_DIR = Path("/root/streamlit_app")

image = modal.Image.debian_slim("3.11").pip_install_from_requirements(
    str(STREAMLIT_DIR / "requirements.txt")
)

app = modal.App(name="EssentialMixerGUI", image=image)


@app.function(
    allow_concurrent_inputs=100,
    mounts=[
        modal.Mount.from_local_dir(
            STREAMLIT_DIR,
            remote_path=REMOTE_STREAMLIT_DIR,
        ),
    ],
)
@modal.web_server(8000)
def deploy_frontend():
    target = shlex.quote(str(REMOTE_STREAMLIT_DIR / "app.py"))
    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false --theme.base=light"
    subprocess.Popen(cmd, shell=True)
