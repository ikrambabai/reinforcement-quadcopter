python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install --upgrade pip
python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl
python notebook Quadcopter_Project.ipynb
