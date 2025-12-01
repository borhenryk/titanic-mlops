#!/usr/bin/env python
import argparse
import yaml
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput
from mlflow.tracking import MlflowClient

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', '-e', required=True)
    parser.add_argument('--model-version', '-v')
    args = parser.parse_args()
    
    with open(f"config/{args.environment}.yaml") as f:
        config = yaml.safe_load(f)
    
    client = MlflowClient()
    full_model_name = f"{config['catalog']}.{config['schema']}.{config['model_name']}"
    
    if args.model_version:
        version = args.model_version
    else:
        try:
            version = client.get_model_version_by_alias(full_model_name, "Champion").version
        except:
            versions = client.search_model_versions(f"name='{full_model_name}'")
            version = max(versions, key=lambda x: int(x.version)).version
    
    w = WorkspaceClient()
    endpoint_name = config['endpoint_name']
    
    try:
        w.serving_endpoints.get(endpoint_name)
        exists = True
    except:
        exists = False
    
    entity = ServedEntityInput(entity_name=full_model_name, entity_version=str(version),
        workload_size="Small", scale_to_zero_enabled=True)
    
    if exists:
        w.serving_endpoints.update_config_and_wait(name=endpoint_name, served_entities=[entity])
    else:
        w.serving_endpoints.create_and_wait(name=endpoint_name,
            config=EndpointCoreConfigInput(served_entities=[entity]))
    
    print(f"Deployed {full_model_name} v{version} to {endpoint_name}")

if __name__ == "__main__":
    main()
