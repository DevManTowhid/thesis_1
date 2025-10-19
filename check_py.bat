for /f %%i in ('dir /s /b *.py') do python -m py_compile "%%i"
