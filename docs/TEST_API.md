# API Testing Commands

## Quick Test (One-liner)

Run this PowerShell command to test the API:

```powershell
curl.exe -X POST "http://localhost:8000/api/tts/upload" -F "text=Hi there [clear throat]..., this is Chris... Do you have a sec? [sniff] ... I really need 400 row-bucks [cough] ... added to my row-blocks account." -F "audio_file=@../20secondchris.wav" -F "temperature=0.8" -F "top_p=0.95" -F "top_k=1000" -F "repetition_penalty=1.2" -F "min_p=0.0" -F "norm_loudness=true" --output test_output.wav
```

## Using the Script

Run the simple test script:

```powershell
.\test_api_simple.ps1
```

Or the more detailed version:

```powershell
.\test_api.ps1
```

## JSON API Test (without file upload)

If you want to test the JSON endpoint instead (requires audio_prompt_path to be set on server):

```powershell
$body = @{
    text = "Hi there [clear throat]..., this is Chris... Do you have a sec? [sniff] ... I really need 400 row-bucks [cough] ... added to my row-blocks account."
    temperature = 0.8
    top_p = 0.95
    top_k = 1000
    repetition_penalty = 1.2
    min_p = 0.0
    norm_loudness = $true
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/api/tts" -Method Post -Body $body -ContentType "application/json" -OutFile "test_output.wav"
```

## Health Check

Test if the server is running:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/health"
```

