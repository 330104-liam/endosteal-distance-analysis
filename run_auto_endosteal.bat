@echo off
setlocal EnableExtensions DisableDelayedExpansion
if "%~1"=="" goto usage
if not "%~3"=="" goto usage
if not exist "%~1" (
  echo Input not found: "%~1"
  pause
  exit /b 2
)
if not "%~2"=="" if not exist "%~2" (
  echo Manifest not found: "%~2"
  pause
  exit /b 2
)
set "PYTHON_EXE="
if defined ENDOSTEAL_PYTHON set "PYTHON_EXE=%ENDOSTEAL_PYTHON%"
if not defined PYTHON_EXE if exist "%~dp0.venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not defined PYTHON_EXE for /f "delims=" %%I in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
if not defined PYTHON_EXE for /f "delims=" %%I in ('python -c "import sys; print(sys.executable)" 2^>nul') do if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
if not defined PYTHON_EXE (
  echo Python 3 not found. See README.md for setup.
  pause
  exit /b 2
)
if not exist "%PYTHON_EXE%" (
  echo Python not found: "%PYTHON_EXE%"
  pause
  exit /b 2
)
set "SCRIPT=%~dp0auto_endosteal_batch.py"
set "ROI_LENGTH_UM=3000"
if exist "%~1\roi_length_um.txt" set /p ROI_LENGTH_UM=<"%~1\roi_length_um.txt"
set "ROI_INPUT="
set /p "ROI_INPUT=ROI length in um [Enter=default %ROI_LENGTH_UM%]: "
if defined ROI_INPUT set "ROI_LENGTH_UM=%ROI_INPUT%"
"%PYTHON_EXE%" -c "import os,sys; s=os.environ.get('ROI_LENGTH_UM',''); sys.exit(0 if s.isascii() and s.isdigit() and int(s)>0 else 2)" >nul 2>&1
if errorlevel 1 (
  echo Invalid ROI length. Enter a positive whole number in um.
  pause
  exit /b 2
)
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss_fff"') do set "RUN_ID=%%I"
if not defined RUN_ID (
  echo Could not create run ID.
  pause
  exit /b 2
)
set "OUTPUT=%~dp0runs\%RUN_ID%_%RANDOM%"
echo Input: "%~1"
echo Output: "%OUTPUT%"
echo Python: "%PYTHON_EXE%"
echo Channels: DAPI=0, Tomato=1, CD31=2
echo ROI length: %ROI_LENGTH_UM% um
set "MANIFEST=%~2"
if not defined MANIFEST if exist "%~1\manifest.csv" set "MANIFEST=%~1\manifest.csv"
if not defined MANIFEST goto no_manifest
echo Manifest: "%MANIFEST%"
"%PYTHON_EXE%" "%SCRIPT%" "%~1" --output "%OUTPUT%" --dapi-channel 0 --red-channel 1 --cd31-channel 2 --roi-length-um "%ROI_LENGTH_UM%" --manifest "%MANIFEST%"
goto done
:no_manifest
"%PYTHON_EXE%" "%SCRIPT%" "%~1" --output "%OUTPUT%" --dapi-channel 0 --red-channel 1 --cd31-channel 2 --roi-length-um "%ROI_LENGTH_UM%"
:done
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" (
  echo Done. Review every AUTO_QC.png before interpreting results.
  echo Results: "%OUTPUT%"
) else (
  echo Analysis failed with exit code %RESULT%. Review the messages above.
)
pause
exit /b %RESULT%
:usage
echo Drag one original three-channel TIFF or a folder of TIFFs onto this BAT file.
echo Optional: put manifest.csv inside the TIFF folder for animal-level tests.
pause
exit /b 2