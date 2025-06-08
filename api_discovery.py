import os
import re
import ast
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class HttpMethod(Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


@dataclass
class Parameter:
    name: str
    in_: str  # path, query, header, cookie
    type_: str
    required: bool = False
    description: str = ""


@dataclass
class Endpoint:
    path: str
    method: HttpMethod
    operation_id: str
    summary: str = ""
    description: str = ""
    parameters: List[Parameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    consumes: List[str] = field(default_factory=list)
    produces: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class APIResource:
    class_name: str
    base_path: str
    endpoints: List[Endpoint] = field(default_factory=list)
    description: str = ""


class JaxRSAnalyzer:
    """Analyzes Java files to discover JAX-RS REST endpoints"""
    
    # JAX-RS annotations
    JAX_RS_ANNOTATIONS = {
        '@Path': 'path',
        '@GET': HttpMethod.GET,
        '@POST': HttpMethod.POST,
        '@PUT': HttpMethod.PUT,
        '@DELETE': HttpMethod.DELETE,
        '@HEAD': HttpMethod.HEAD,
        '@OPTIONS': HttpMethod.OPTIONS,
        '@PATCH': HttpMethod.PATCH,
        '@Consumes': 'consumes',
        '@Produces': 'produces',
        '@PathParam': 'path',
        '@QueryParam': 'query',
        '@HeaderParam': 'header',
        '@CookieParam': 'cookie',
        '@FormParam': 'formData'
    }
    
    def __init__(self, repo_path: str):
        self.repo_path = Path(repo_path)
        self.resources: List[APIResource] = []
    
    def analyze(self) -> List[APIResource]:
        """Main method to analyze the repository"""
        java_files = self._find_java_files()
        
        for java_file in java_files:
            resource = self._analyze_java_file(java_file)
            if resource and resource.endpoints:
                self.resources.append(resource)
        
        return self.resources
    
    def _find_java_files(self) -> List[Path]:
        """Find all Java files in the repository"""
        return list(self.repo_path.rglob("*.java"))
    
    def _analyze_java_file(self, file_path: Path) -> Optional[APIResource]:
        """Analyze a single Java file for JAX-RS endpoints"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if this is a JAX-RS resource class
            if not self._is_jaxrs_resource(content):
                return None
            
            # Extract class-level information
            class_name = self._extract_class_name(content)
            base_path = self._extract_class_path(content)
            
            resource = APIResource(
                class_name=class_name,
                base_path=base_path or ""
            )
            
            # Extract method-level endpoints
            endpoints = self._extract_endpoints(content, base_path)
            resource.endpoints = endpoints
            
            return resource
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _is_jaxrs_resource(self, content: str) -> bool:
        """Check if the file contains JAX-RS annotations"""
        jaxrs_patterns = [
            r'@Path\s*\(',
            r'@GET[\s\n]',
            r'@POST[\s\n]',
            r'@PUT[\s\n]',
            r'@DELETE[\s\n]'
        ]
        return any(re.search(pattern, content) for pattern in jaxrs_patterns)
    
    def _extract_class_name(self, content: str) -> str:
        """Extract the class name from Java file"""
        match = re.search(r'public\s+class\s+(\w+)', content)
        return match.group(1) if match else "Unknown"
    
    def _extract_class_path(self, content: str) -> Optional[str]:
        """Extract class-level @Path annotation"""
        # Look for class-level @Path
        class_pattern = r'@Path\s*\(\s*["\']([^"\']+)["\']\s*\)\s*(?:public\s+)?class'
        match = re.search(class_pattern, content)
        if match:
            return match.group(1)
        
        # Also check for @Path before the class declaration
        path_pattern = r'@Path\s*\(\s*["\']([^"\']+)["\']\s*\)'
        class_decl_pattern = r'public\s+class\s+\w+'
        
        path_match = re.search(path_pattern, content)
        if path_match:
            path_pos = path_match.end()
            class_match = re.search(class_decl_pattern, content[path_pos:])
            if class_match and class_match.start() < 100:  # Path should be close to class
                return path_match.group(1)
        
        return None
    
    def _extract_endpoints(self, content: str, base_path: str) -> List[Endpoint]:
        """Extract all endpoints from the Java file"""
        endpoints = []
        
        # Find all methods with HTTP annotations
        method_pattern = r'(@\w+\s*(?:\([^)]*\))?\s*)*\s*public\s+\w+\s+(\w+)\s*\([^)]*\)'
        
        for match in re.finditer(method_pattern, content):
            annotations = match.group(1) or ""
            method_name = match.group(2)
            
            # Check if this method has HTTP method annotations
            http_method = self._extract_http_method(annotations)
            if not http_method:
                continue
            
            # Extract method-level path
            method_path = self._extract_method_path(annotations)
            
            # Combine base path and method path
            full_path = self._combine_paths(base_path, method_path)
            
            # Extract parameters
            parameters = self._extract_parameters(match.group(0), content)
            
            # Extract consumes/produces
            consumes = self._extract_media_types(annotations, '@Consumes')
            produces = self._extract_media_types(annotations, '@Produces')
            
            endpoint = Endpoint(
                path=full_path,
                method=http_method,
                operation_id=f"{method_name}",
                parameters=parameters,
                consumes=consumes or ["application/json"],
                produces=produces or ["application/json"]
            )
            
            endpoints.append(endpoint)
        
        return endpoints
    
    def _extract_http_method(self, annotations: str) -> Optional[HttpMethod]:
        """Extract HTTP method from annotations"""
        for annotation, method in self.JAX_RS_ANNOTATIONS.items():
            if isinstance(method, HttpMethod) and annotation in annotations:
                return method
        return None
    
    def _extract_method_path(self, annotations: str) -> str:
        """Extract method-level @Path annotation"""
        match = re.search(r'@Path\s*\(\s*["\']([^"\']+)["\']\s*\)', annotations)
        return match.group(1) if match else ""
    
    def _extract_media_types(self, annotations: str, annotation_type: str) -> List[str]:
        """Extract media types from @Consumes or @Produces"""
        pattern = rf'{annotation_type}\s*\(\s*\{{?\s*([^)]+)\s*\}}?\s*\)'
        match = re.search(pattern, annotations)
        if match:
            media_types = match.group(1)
            # Extract individual media types
            types = re.findall(r'["\']([^"\']+)["\']', media_types)
            return types
        return []
    
    def _extract_parameters(self, method_signature: str, content: str) -> List[Parameter]:
        """Extract parameters from method signature"""
        parameters = []
        
        # Extract method parameters
        param_pattern = r'@(\w+Param)\s*\(\s*["\']([^"\']+)["\']\s*\)\s+(\w+)\s+(\w+)'
        
        for match in re.finditer(param_pattern, method_signature):
            param_type = match.group(1)
            param_name = match.group(2)
            java_type = match.group(3)
            var_name = match.group(4)
            
            if param_type in ['PathParam', 'QueryParam', 'HeaderParam', 'CookieParam']:
                in_ = self.JAX_RS_ANNOTATIONS[f'@{param_type}']
                
                # Map Java types to OpenAPI types
                openapi_type = self._map_java_to_openapi_type(java_type)
                
                parameter = Parameter(
                    name=param_name,
                    in_=in_,
                    type_=openapi_type,
                    required=(in_ == 'path')  # Path params are always required
                )
                parameters.append(parameter)
        
        return parameters
    
    def _map_java_to_openapi_type(self, java_type: str) -> str:
        """Map Java types to OpenAPI types"""
        type_mapping = {
            'String': 'string',
            'int': 'integer',
            'Integer': 'integer',
            'long': 'integer',
            'Long': 'integer',
            'float': 'number',
            'Float': 'number',
            'double': 'number',
            'Double': 'number',
            'boolean': 'boolean',
            'Boolean': 'boolean',
            'Date': 'string',
            'LocalDate': 'string',
            'LocalDateTime': 'string'
        }
        return type_mapping.get(java_type, 'string')
    
    def _combine_paths(self, base_path: str, method_path: str) -> str:
        """Combine base path and method path"""
        if not base_path:
            return method_path or "/"
        if not method_path:
            return base_path
        
        # Ensure proper path combination
        base = base_path.rstrip('/')
        method = method_path.lstrip('/')
        
        return f"{base}/{method}"


class OpenAPIGenerator:
    """Generates OpenAPI 3.0 specification from discovered APIs"""
    
    def __init__(self, api_resources: List[APIResource]):
        self.api_resources = api_resources
    
    def generate(self, title: str = "Discovered API", version: str = "1.0.0") -> Dict[str, Any]:
        """Generate OpenAPI specification"""
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": "Auto-discovered API from JAX-RS annotations"
            },
            "servers": [
                {
                    "url": "http://localhost:8080",
                    "description": "Local development server"
                }
            ],
            "paths": {}
        }
        
        # Generate paths from resources
        for resource in self.api_resources:
            for endpoint in resource.endpoints:
                path = endpoint.path
                method = endpoint.method.value.lower()
                
                if path not in openapi_spec["paths"]:
                    openapi_spec["paths"][path] = {}
                
                operation = {
                    "operationId": endpoint.operation_id,
                    "summary": endpoint.summary or f"{endpoint.method.value} {path}",
                    "tags": [resource.class_name],
                    "responses": {
                        "200": {
                            "description": "Successful response",
                            "content": {
                                media_type: {"schema": {"type": "object"}}
                                for media_type in endpoint.produces
                            }
                        }
                    }
                }
                
                # Add parameters
                if endpoint.parameters:
                    operation["parameters"] = []
                    for param in endpoint.parameters:
                        operation["parameters"].append({
                            "name": param.name,
                            "in": param.in_,
                            "required": param.required,
                            "schema": {
                                "type": param.type_
                            }
                        })
                
                # Add request body for POST/PUT
                if endpoint.method in [HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH]:
                    if endpoint.consumes:
                        operation["requestBody"] = {
                            "content": {
                                media_type: {"schema": {"type": "object"}}
                                for media_type in endpoint.consumes
                            }
                        }
                
                openapi_spec["paths"][path][method] = operation
        
        return openapi_spec
    
    def save_to_file(self, spec: Dict[str, Any], output_path: str):
        """Save OpenAPI spec to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(spec, f, default_flow_style=False, sort_keys=False)


class BackstageGenerator:
    """Generates Backstage catalog YAML from discovered APIs"""
    
    def __init__(self, api_resources: List[APIResource]):
        self.api_resources = api_resources
    
    def generate(self, api_name: str, openapi_path: str) -> Dict[str, Any]:
        """Generate Backstage catalog entry"""
        catalog_entry = {
            "apiVersion": "backstage.io/v1alpha1",
            "kind": "API",
            "metadata": {
                "name": api_name.lower().replace(" ", "-"),
                "title": api_name,
                "description": "Auto-discovered JAX-RS API",
                "annotations": {
                    "backstage.io/source-location": "url:https://github.com/your-org/your-repo"
                }
            },
            "spec": {
                "type": "openapi",
                "lifecycle": "production",
                "owner": "team-unknown",
                "system": "unknown-system",
                "definition": f"$text: ./{openapi_path}"
            }
        }
        
        return catalog_entry
    
    def save_to_file(self, catalog: Dict[str, Any], output_path: str):
        """Save Backstage catalog to YAML file"""
        with open(output_path, 'w') as f:
            yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)


def main(repo_path: str, output_dir: str = "./output"):
    """Main function to run the API discovery tool"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing repository: {repo_path}")
    
    # Step 1: Analyze repository
    analyzer = JaxRSAnalyzer(repo_path)
    api_resources = analyzer.analyze()
    
    if not api_resources:
        print("No JAX-RS APIs found in the repository")
        return
    
    print(f"Found {len(api_resources)} API resources")
    for resource in api_resources:
        print(f"  - {resource.class_name}: {len(resource.endpoints)} endpoints")
    
    # Step 2: Generate OpenAPI spec
    openapi_gen = OpenAPIGenerator(api_resources)
    openapi_spec = openapi_gen.generate(
        title="My JAX-RS API",
        version="1.0.0"
    )
    
    openapi_path = os.path.join(output_dir, "openapi.yaml")
    openapi_gen.save_to_file(openapi_spec, openapi_path)
    print(f"Generated OpenAPI spec: {openapi_path}")
    
    # Step 3: Generate Backstage catalog
    backstage_gen = BackstageGenerator(api_resources)
    catalog = backstage_gen.generate("My JAX-RS API", "openapi.yaml")
    
    catalog_path = os.path.join(output_dir, "catalog-info.yaml")
    backstage_gen.save_to_file(catalog, catalog_path)
    print(f"Generated Backstage catalog: {catalog_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python api_discovery.py <repository_path> [output_directory]")
        print("Example: python api_discovery.py ./my-java-repo ./output")
        sys.exit(1)
    
    repo_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./output"
    
    # Validate repository path
    if not os.path.exists(repo_path):
        print(f"Error: Repository path '{repo_path}' does not exist")
        sys.exit(1)
    
    if not os.path.isdir(repo_path):
        print(f"Error: Repository path '{repo_path}' is not a directory")
        sys.exit(1)
    
    main(repo_path, output_dir)