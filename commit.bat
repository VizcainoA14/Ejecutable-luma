cd C:\Users\adria\Documents\GitHub\LumaWeb

setlocal enabledelayedexpansion

rem Obtener la fecha y la hora actual
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set datetime=%%i

rem Extraer año, mes, día, hora y minutos
set year=!datetime:~0,4!
set month=!datetime:~4,2!
set day=!datetime:~6,2!
set hour=!datetime:~8,2!
set minute=!datetime:~10,2!

rem Imprimir la fecha y hora en el formato deseado
git add --all
git commit -m "autoCommit DATA %day%/%month%/%year% %hour%:%minute%"
git push
exit

