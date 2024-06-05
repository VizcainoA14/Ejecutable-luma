@ECHO OFF

CHCP 65001 > NUL
cd C:\Users\adria\Desktop\Saiyayin
call .venv\Scripts\activate
python geturls.py

exit

PAUSE > NUL