#!/usr/bin/env cwl-runner
cwlVersion: v1.2
class: CommandLineTool

hints:
  DockerRequirement:
    dockerPull: hariszaf/fetch-tool:latest

requirements:
  NetworkAccess:
     networkAccess: true

baseCommand: [ get_raw_data.sh ]

inputs: 

 run_accession_number: 
    type: string
    label: The accession number of the ENA run to be downloaded. 
    inputBinding: 
        prefix: -r

 private_data: 
    type: boolean
    label: In case the ENA data are still private, you need to denote that and fill your credentials in the Configuration file.
    default: false
    inputBinding:
        prefix: -p 

 ena_api_username: 
    type: string
    label: The username for the ENA account to be accessed to get raw data.
    default: ""
    inputBinding: 
        prefix: -u

 ena_api_password:
    type: string
    label: The password for the ENA account to be accessed to get raw data.
    default: ""
    inputBinding: 
        prefix: -k

outputs:
    raw_data:
        type: Directory[]
        outputBinding:
            glob: raw_data_from_ENA
