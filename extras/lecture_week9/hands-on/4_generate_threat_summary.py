# Filename: 4_generate_threat_summary.py
import json
import asyncio
import aiohttp
import os
import sys


async def generate_summary(ioc_data, api_key):
    """
    Sends IoC data to the Gemini API and returns a generated threat summary.
    """
    prompt = f"""
    As a senior cyber threat intelligence analyst, your task is to synthesize the following structured IoC data into a brief, human-readable intelligence report.

    The report must contain two sections:
    1.  **Threat Summary:** A one-paragraph narrative explaining the nature of the threat, the likely actor, and the malware involved.
    2.  **Recommended Actions:** A numbered list of 3-5 prioritized actions for a Security Operations Center (SOC) to take immediately.

    IoC Data:
    {json.dumps(ioc_data, indent=2)}
    """

    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(apiUrl, json=payload) as response:
                result = await response.json()

                if response.status != 200:
                    return f"Error: API returned status {response.status}. Response: {json.dumps(result)}"

                if result.get('candidates'):
                    return result['candidates'][0]['content']['parts'][0]['text']
                else:
                    return "Error: Could not generate summary. Full API Response: " + json.dumps(result)

    except aiohttp.ClientConnectorError as e:
        return f"An error occurred: Could not connect to the API endpoint. {e}"
    except Exception as e:
        return f"An error occurred: {e}"


# Data representing Indicators of Compromise
structured_iocs = {
    "malware_families": ["Guloader", "Cobalt Strike"],
    "threat_actors": ["APT42", "Fancy Bear"],
    "cves": ["CVE-2021-44228"],
    "ips": ["185.191.207.57"],
    "hashes": ["275a021b7cf35a0b943505c61988cc05"]
}


async def main():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("---")
        print("🚨 Error: GOOGLE_API_KEY environment variable not set.")
        print("To run this script, you need to set your API key.")
        print("\nFor Linux/macOS, use:\n  export GOOGLE_API_KEY='YOUR_API_KEY_HERE'")
        print("\nFor Windows (PowerShell), use:\n  $env:GOOGLE_API_KEY=\"YOUR_API_KEY_HERE\"")
        print("\nReplace 'YOUR_API_KEY_HERE' with the key you obtained from Google AI Studio.")
        print("---")
        sys.exit(1)

    # --- Displaying context and input data ---
    print("---")
    print("Context: Generating a threat intelligence report from structured IoCs.")
    print("Input being sent to Gemini:")
    print(json.dumps(structured_iocs, indent=2))
    print("--------------------------------------------------")
    print("\nGenerating response...\n")
    # -------------------------------------------

    summary = await generate_summary(structured_iocs, api_key)
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())

