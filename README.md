# EEG-Analysis


## Using virtualenv
> Instaling

```cmd
python -m pip install --user virtualenv
```
> Creating new env

```cmd
python -m venv env_tcc_eeg

python -m venv env_tcc_eegv2
```
> Activating env

```cmd
.\env_tcc_eeg\Scripts\activate

.\env_tcc_eegv2\Scripts\activate
```
> Leaving virtual env

```cmd
deactivate
```

> Updating pip

```cmd
python -m pip install --upgrade pip
```

> Installing packages

```cmd
python -m pip install -r requirements.txt

python -m pip install -r requirementsV2.txt
```

https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/

https://pypi.org/project/Braindecode/#history