@echo off
echo.
echo ======================================
echo   Iniciando Jupyter con GPU Support
echo ======================================
echo.

REM Activar entorno virtual
call venv_gpu\Scripts\activate.bat

REM Verificar GPU
echo Verificando GPU...
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No detectada\"}')"
echo.

REM Iniciar Jupyter
echo Iniciando Jupyter Lab...
jupyter lab

pause

