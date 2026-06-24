Dim WinScriptHost
Set WinScriptHost = CreateObject("WScript.Shell")

' 1. 가상환경 활성화 후 FastAPI 서버 백그라운드 구동 (80번 포트)
WinScriptHost.Run "cmd /c ""C:\python_1st_document\aiml\.venv\Scripts\activate.bat"" && uvicorn aismartfarm:create_app --factory --host 0.0.0.0 --port 80", 0, False

' 2. [교정] 터널 실행 경로와 파일 위치를 절대 경로로 확실하게 묶어서 수동 실행
WinScriptHost.Run "cmd /c cd /d C:\python_1st_document\aiml\.cloudflared && C:\python_1st_document\aiml\.cloudflared\cloudflared.exe --config C:\python_1st_document\aiml\.cloudflared\config.yml tunnel run paprika-tunnel", 0, False

Set WinScriptHost = Nothing