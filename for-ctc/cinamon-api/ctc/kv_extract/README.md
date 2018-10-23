
├── Makefile                <- tasks
├── config.yml              <- config file in YAML, can be exported as env vars if needed
├── config-private.yml      <- config file with private config (password, api keys, etc.)
script
	--- set_env
	--- run
├── data
│   └── raw
│   ├── intermediate
│   ├── processed
│   ├── temp
├── results
│   ├── outputs
│   ├── models
├── documents
│   ├── docs
│   ├── images
│   └── references
├── notebooks               <- notebooks for explorations / prototyping
└── src                     <- all source code, internal org as needed
--- cache 		    <- store all temporary file during process/ can be clean after 
