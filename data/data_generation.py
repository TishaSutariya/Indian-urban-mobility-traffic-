import requests
import pandas as pd
import os
import time

# ============================================================
# INDIAN URBAN MOBILITY - DATA COLLECTION
# Official Government of India data.gov.in resources
# ============================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 70)
print("INDIAN URBAN MOBILITY - DATA COLLECTION")
print("=" * 70)

# ------------------------------------------------------------
# Official resource pages
# ------------------------------------------------------------

RESOURCE_PAGES = {
    "state_accidents":
        "https://www.data.gov.in/resource/"
        "stateut-wise-total-number-road-accidents-india-2016-2019",

    "collision":
        "https://punjab.data.gov.in/resource/"
        "stateut-wise-accidents-classified-according-type-collision-during-2019",

    "weather":
        "https://up.data.gov.in/resource/"
        "stateut-wise-accidents-classified-according-type-weather-condition-during-2019",
}

print("\nOfficial Government sources selected:")
print("-" * 70)

for name, url in RESOURCE_PAGES.items():
    print(f"{name}:")
    print(url)

# ------------------------------------------------------------
# Save source URLs for project documentation
# ------------------------------------------------------------

sources_file = os.path.join(
    OUTPUT_DIR,
    "data_sources.txt"
)

with open(
    sources_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "INDIAN TRAFFIC DATA SOURCES\n"
        "============================\n\n"
    )

    for name, url in RESOURCE_PAGES.items():

        f.write(
            f"{name}\n"
            f"{url}\n\n"
        )

print("\nSource list saved:")
print(sources_file)

# ------------------------------------------------------------
# Try downloading resource pages
# ------------------------------------------------------------

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/140 Safari/537.36"
})

for name, url in RESOURCE_PAGES.items():

    print("\n" + "=" * 70)
    print(f"CHECKING: {name}")
    print("=" * 70)

    try:

        response = session.get(
            url,
            timeout=30
        )

        print(
            "HTTP status:",
            response.status_code
        )

        if response.status_code == 200:

            print(
                "Official resource page reached successfully."
            )

            # Save the HTML for inspection
            html_file = os.path.join(
                OUTPUT_DIR,
                f"{name}_source.html"
            )

            with open(
                html_file,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    response.text
                )

            print(
                "Saved source page:"
            )

            print(
                html_file
            )

        else:

            print(
                "Could not access resource."
            )

    except requests.exceptions.Timeout:

        print(
            "Request timed out."
        )

    except requests.exceptions.RequestException as e:

        print(
            "Request failed:"
        )

        print(
            e
        )

    time.sleep(2)

print("\n" + "=" * 70)
print("DATA COLLECTION CHECK FINISHED")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "The official data.gov.in website does not expose "
    "every CSV through a simple static URL."
)

print(
    "Therefore this script saves the official resource "
    "references instead of pretending that a CSV was downloaded."
)

print(
    "\nNext step: use the official resource/API/download "
    "endpoint to create the actual CSV files."
)