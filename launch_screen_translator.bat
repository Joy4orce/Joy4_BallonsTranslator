@REM Screen translator entry point — Yomichan-style overlay translation
@REM for images on screen. Reuses BallonsTranslator's text detection,
@REM manga OCR, and translator modules.

cd %~dp0

@echo off

set PATH=ballontrans_pylibs_win;ballontrans_pylibs_win\Scripts;PortableGit\cmd;%PATH%
set PYTHON=python.exe

set ERROR_REPORTING=FALSE

mkdir tmp 2>NUL

%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Couldn't launch python
goto :show_stdout_stderr


:launch
%PYTHON% -m tools.screen_translator.main %*
pause
exit /b


:show_stdout_stderr
echo.
echo exit code: %errorlevel%
echo.
type tmp\stdout.txt
echo.
type tmp\stderr.txt
echo.
echo Launch unsuccessful. Exiting.
pause
