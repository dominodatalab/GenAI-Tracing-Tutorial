#!/bin/bash

# Domino App launch script for TriageFlow Configuration App
# Configures Streamlit to run on port 8888 as required by Domino

mkdir -p ~/.streamlit

cat << EOF > ~/.streamlit/config.toml
[browser]
gatherUsageStats = false

[server]
port = 8888
enableCORS = false
enableXsrfProtection = false
address = "0.0.0.0"
headless = true

[theme]
base = "light"
EOF

streamlit run /mnt/code/app/main.py
