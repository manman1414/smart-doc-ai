' SmartDoc AI 一键启动
' Author: Cursor Agent / 2026-07-06

Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
ps1 = dir & "\start-all.ps1"
psExe = sh.ExpandEnvironmentStrings("%SystemRoot%") & "\System32\WindowsPowerShell\v1.0\powershell.exe"

If Not fso.FileExists(ps1) Then
  MsgBox "找不到 start-all.ps1", vbCritical, "SmartDoc AI"
  WScript.Quit 1
End If

sh.CurrentDirectory = dir
sh.Run """" & psExe & """ -NoProfile -NoExit -ExecutionPolicy Bypass -File """ & ps1 & """", 1, False
