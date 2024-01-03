# Reproducing AlexNet Paper

This repository is the official implementation of Reproducing AlexNet Paper as the final project of ARI5004 Deep Learning course.

## Environment

In this project Python virtual environment has been utilized. To create a virtual environment for the project run the following code:

```virtualenv
python -m venv <path-to-virtualenv>
```

After virtual environment is created to activate the virtual environment run the following code:

```windows
call <path-to-virtualenv>/Scripts/activate.bat
```

```linux&macos
source <path-to-virtualenv>/bin/activate
```

After activating the virtual environment your terminal should look like below:

```terminal-windows
(venv) C:\<path-to-project>
```

```terminal-macos
(venv) machine-name:path-to-project username$
```

```terminal-linux
(venv) username@machine-name:path-to-project$
```

## Requirements

After activating the virtual environment to install the requirements run the following code:

```setup
pip install -r requirements.txt
```

## Training

To train the AlexNet model in the paper, run the following command:

```train
python train.py --root-dir <path-to-dataset> --
```