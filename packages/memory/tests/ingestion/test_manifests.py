"""What a project says it depends on, read from its package manifest.

A manifest is where a project writes down what it needs. Left unread, the
graph knows every `import requests` and nothing about which projects
declare requests, at what version, or only for tests. Read as claims --
quoted from the line, cited to the file, extracted rather than stated --
it answers "what does this project depend on" from the ledger like every
other question, and one package named by five manifests is one thing.
"""

from __future__ import annotations

import pytest

from scone_memory import HashEmbedder, InMemoryDocumentStore, InMemoryVectorIndex, MemoryEngine
from scone_memory.ingestion.code_graph import DEFINES, MAX_CLAIMS, record_claims
from scone_memory.ingestion.manifests import DEPENDS_ON, DEVELOPS_WITH, is_manifest, manifest_claims


def said(claims, predicate=None):
    return [(c.subject, c.predicate, c.object) for c in claims if predicate is None or c.predicate == predicate]


PYPROJECT = '''[build-system]
requires = ["hatchling>=1.20", "hatch-vcs"]

[project]
name = "Scone_Memory"
dependencies = [
    "requests>=2.31",
    "uvicorn[standard]==0.30.1",
    "Beautiful_Soup4 ; python_version < '3.14'",
]

[project.optional-dependencies]
aws = ["boto3>=1.34"]

[dependency-groups]
test = ["pytest>=8", "pytest-asyncio"]
'''


def test_a_pep_621_manifest_declares_its_package_and_what_it_needs():
    claims = manifest_claims(PYPROJECT, "packages/memory/pyproject.toml")
    assert said(claims, DEFINES) == [("packages/memory/pyproject.toml", DEFINES, "scone-memory")]
    assert said(claims, DEPENDS_ON) == [
        ("scone-memory", DEPENDS_ON, "requests"),
        ("scone-memory", DEPENDS_ON, "uvicorn"),
        ("scone-memory", DEPENDS_ON, "beautiful-soup4"),
        ("scone-memory", DEPENDS_ON, "boto3"),
    ], "extras and markers are not part of the name, and names are normalised as the index normalises them"
    assert said(claims, DEVELOPS_WITH) == [
        ("scone-memory", DEVELOPS_WITH, "hatchling"),
        ("scone-memory", DEVELOPS_WITH, "hatch-vcs"),
        ("scone-memory", DEVELOPS_WITH, "pytest"),
        ("scone-memory", DEVELOPS_WITH, "pytest-asyncio"),
    ], "what builds or tests a project is not what it runs on"


def test_each_claim_quotes_the_line_it_was_read_from_and_spans_its_bytes():
    content = "# ünïcode above, so byte and character offsets differ\n" + PYPROJECT
    claims = manifest_claims(content, "pyproject.toml")
    by_object = {c.object: c for c in claims}
    uvicorn = by_object["uvicorn"]
    assert uvicorn.quote == '"uvicorn[standard]==0.30.1",'
    assert uvicorn.first_line == 9
    assert content.encode()[uvicorn.start:uvicorn.end].decode().strip() == uvicorn.quote
    assert by_object["hatchling"].quote == 'requires = ["hatchling>=1.20", "hatch-vcs"]'
    assert by_object["hatch-vcs"].first_line == by_object["hatchling"].first_line, \
        "two dependencies on one line quote that line twice"
    assert by_object["scone-memory"].quote == 'name = "Scone_Memory"'


def test_a_poetry_manifest_reads_its_groups_and_never_claims_python_itself():
    content = '''[tool.poetry]
name = "shelf"

[tool.poetry.dependencies]
python = "^3.11"
httpx = { version = ">=0.27", extras = ["http2"] }

[tool.poetry.dev-dependencies]
black = "*"

[tool.poetry.group.docs.dependencies]
mkdocs = "^1.6"
'''
    claims = manifest_claims(content, "pyproject.toml")
    assert said(claims, DEFINES) == [("pyproject.toml", DEFINES, "shelf")]
    assert said(claims, DEPENDS_ON) == [("shelf", DEPENDS_ON, "httpx")]
    assert said(claims, DEVELOPS_WITH) == [("shelf", DEVELOPS_WITH, "black"), ("shelf", DEVELOPS_WITH, "mkdocs")]


def test_a_requirements_file_reads_names_and_skips_what_is_not_one():
    content = '''# runtime
requests==2.32.3   # pinned
-r base.txt
-e .
--index-url https://example.invalid/simple
Flask[async] >= 3.0
pkg @ https://example.invalid/pkg-1.0.tar.gz

numpy; python_version < "3.13"
'''
    claims = manifest_claims(content, "api/requirements-dev.txt")
    assert said(claims) == [
        ("api/requirements-dev.txt", DEPENDS_ON, "requests"),
        ("api/requirements-dev.txt", DEPENDS_ON, "flask"),
        ("api/requirements-dev.txt", DEPENDS_ON, "pkg"),
        ("api/requirements-dev.txt", DEPENDS_ON, "numpy"),
    ], "a file with no package name speaks for itself by path"
    assert [c.first_line for c in claims] == [2, 6, 7, 9]


def test_a_package_json_separates_what_it_runs_on_from_what_it_develops_with():
    content = '''{
  "name": "@scone/console",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.3.1",
    "@tanstack/react-query": "5.51.1"
  },
  "peerDependencies": { "react-dom": ">=18" },
  "devDependencies": {
    "vitest": "^2.0.0",
    "TypeScript": "5.5.4"
  }
}
'''
    claims = manifest_claims(content, "webapp/package.json")
    assert said(claims, DEFINES) == [("webapp/package.json", DEFINES, "@scone/console")]
    assert said(claims, DEPENDS_ON) == [
        ("@scone/console", DEPENDS_ON, "react"),
        ("@scone/console", DEPENDS_ON, "@tanstack/react-query"),
        ("@scone/console", DEPENDS_ON, "react-dom"),
    ]
    assert said(claims, DEVELOPS_WITH) == [
        ("@scone/console", DEVELOPS_WITH, "vitest"),
        ("@scone/console", DEVELOPS_WITH, "typescript"),
    ]
    by_object = {c.object: c for c in claims}
    assert by_object["react"].quote == '"react": "^18.3.1",'
    assert by_object["react-dom"].first_line == 8


def test_a_cargo_manifest_reads_every_dependency_table_and_the_real_crate_behind_a_rename():
    content = '''[package]
name = "scone_core"
version = "0.1.0"

[dependencies]
serde = { version = "1", features = ["derive"] }
tokio-runtime = { package = "tokio", version = "1.38" }

[dev-dependencies]
proptest = "1"

[build-dependencies]
cc = "1"

[target.'cfg(unix)'.dependencies]
libc = "0.2"
'''
    claims = manifest_claims(content, "crates/core/Cargo.toml")
    assert said(claims, DEFINES) == [("crates/core/Cargo.toml", DEFINES, "scone-core")]
    assert said(claims, DEPENDS_ON) == [
        ("scone-core", DEPENDS_ON, "serde"),
        ("scone-core", DEPENDS_ON, "tokio"),
        ("scone-core", DEPENDS_ON, "libc"),
    ], "a renamed dependency is the crate it names, not the alias"
    assert said(claims, DEVELOPS_WITH) == [("scone-core", DEVELOPS_WITH, "proptest"), ("scone-core", DEVELOPS_WITH, "cc")]


def test_a_cargo_workspace_without_a_package_speaks_by_path():
    content = '''[workspace]
members = ["crates/*"]

[workspace.dependencies]
serde = "1"
'''
    claims = manifest_claims(content, "Cargo.toml")
    assert said(claims) == [("Cargo.toml", DEPENDS_ON, "serde")]


def test_a_go_module_reads_direct_requirements_only():
    content = '''module github.com/scone/serve

go 1.22

require github.com/gorilla/mux v1.8.1

require (
\tgolang.org/x/net v0.27.0
\tgithub.com/stretchr/testify v1.9.0 // indirect
)
'''
    claims = manifest_claims(content, "serve/go.mod")
    assert said(claims, DEFINES) == [("serve/go.mod", DEFINES, "github.com/scone/serve")]
    assert said(claims, DEPENDS_ON) == [
        ("github.com/scone/serve", DEPENDS_ON, "github.com/gorilla/mux"),
        ("github.com/scone/serve", DEPENDS_ON, "golang.org/x/net"),
    ], "an indirect requirement is what a dependency needs, not what this module declares"
    assert [c.first_line for c in claims if c.predicate == DEPENDS_ON] == [5, 8]


@pytest.mark.parametrize("path,expected", [
    ("pyproject.toml", True), ("packages/memory/PyProject.toml", True), ("requirements.txt", True),
    ("api/requirements-dev.txt", True), ("requirements/test.txt", True), ("package.json", True),
    ("crates/core/Cargo.toml", True), ("go.mod", True),
    ("pom.xml", True), ("app/build.gradle", True), ("app/build.gradle.kts", True), ("Gemfile", True),
    ("composer.json", True), ("Pipfile", True), ("src/App/App.csproj", True), ("Lib.fsproj", True),
    ("notes.toml", False), ("tsconfig.json", False), ("go.sum", False), ("requirements.md", False),
    ("package-lock.json", False), ("Gemfile.lock", False), ("settings.gradle", False), ("composer.lock", False),
    ("App.csproj.user", False), ("", False),
])
def test_a_manifest_is_known_by_its_name_wherever_it_sits(path, expected):
    assert is_manifest(path) is expected


@pytest.mark.parametrize("path,content", [
    ("pyproject.toml", "[project\nname = 'broken'"),
    ("package.json", '{"name": "x", "dependencies": ['),
    ("package.json", '["not", "an", "object"]'),
    ("Cargo.toml", "[dependencies]\nserde = 1"),
    ("pom.xml", "<project><dependencies>"), ("pom.xml", "<notes/>"), ("composer.json", "[1, 2]"),
    ("App.csproj", "<Project><ItemGroup>"), ("Pipfile", "[packages\nx = '*'"),
])
def test_a_manifest_that_does_not_parse_claims_nothing_and_never_raises(path, content):
    assert manifest_claims(content, path) == ()


def test_a_dependency_list_of_the_wrong_shape_is_skipped_and_the_name_still_stands():
    claims = manifest_claims('[project]\nname = "x"\ndependencies = "not a list"', "pyproject.toml")
    assert said(claims) == [("pyproject.toml", DEFINES, "x")]


def test_a_file_that_is_not_a_manifest_claims_nothing():
    assert manifest_claims(PYPROJECT, "notes.toml") == ()
    assert manifest_claims("", "pyproject.toml") == ()


def test_the_claim_cap_holds_for_manifests_too():
    content = "\n".join(f"pkg{n}==1.0" for n in range(MAX_CLAIMS + 5))
    assert len(manifest_claims(content, "requirements.txt")) == MAX_CLAIMS


@pytest.mark.asyncio
async def test_a_remembered_manifest_is_recorded_as_extracted_quoted_claims():
    engine = await MemoryEngine(InMemoryDocumentStore(), InMemoryVectorIndex(), HashEmbedder()).open()
    try:
        added = await engine.remember("s", PYPROJECT, kind="file", source="packages/memory/pyproject.toml")
        recorded: list = []
        count = await record_claims(engine, "s", episode_id=added.episode_id, content=PYPROJECT,
                                    path="packages/memory/pyproject.toml", when=engine.clock(), _recorded=recorded)
        assert count == 9 and len(recorded) == 9
        facts = await engine.facts("s")
        depends = sorted(f.object for f in facts if f.predicate == DEPENDS_ON)
        assert depends == ["beautiful-soup4", "boto3", "requests", "uvicorn"]
        requests = next(f for f in facts if f.object == "requests")
        assert (requests.origin, requests.source_episode_id, requests.quote) == \
            ("extracted", added.episode_id, '"requests>=2.31",')
        assert requests.grounded is True
    finally:
        await engine.close()


POM = """<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>org.example</groupId>
    <artifactId>parent</artifactId>
    <version>1</version>
  </parent>
  <artifactId>Service</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
      <version>2.17.0</version>
    </dependency>
    <dependency>
      <groupId>org.junit.jupiter</groupId>
      <artifactId>junit-jupiter</artifactId>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.slf4j</groupId>
      <artifactId>slf4j-api</artifactId>
      <scope>provided</scope>
    </dependency>
  </dependencies>
  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
      </plugin>
    </plugins>
  </build>
</project>
"""


def test_a_maven_pom_reads_coordinates_scopes_and_plugins_under_the_parents_group():
    claims = manifest_claims(POM, "service/pom.xml")
    assert said(claims, DEFINES) == [("service/pom.xml", DEFINES, "org.example:service")], "the group is the parent's"
    assert said(claims, DEPENDS_ON) == [("org.example:service", DEPENDS_ON, "com.fasterxml.jackson.core:jackson-databind"),
                                        ("org.example:service", DEPENDS_ON, "org.slf4j:slf4j-api")]
    assert said(claims, DEVELOPS_WITH) == [("org.example:service", DEVELOPS_WITH, "org.junit.jupiter:junit-jupiter"),
                                           ("org.example:service", DEVELOPS_WITH, "org.apache.maven.plugins:maven-surefire-plugin")]
    by_object = {c.object: c for c in claims}
    assert by_object["org.junit.jupiter:junit-jupiter"].quote == "<artifactId>junit-jupiter</artifactId>"
    assert by_object["org.junit.jupiter:junit-jupiter"].first_line == 18
    assert POM.encode()[by_object["org.slf4j:slf4j-api"].start:by_object["org.slf4j:slf4j-api"].end].decode().strip() \
        == "<artifactId>slf4j-api</artifactId>"


def test_a_gradle_build_reads_groovy_and_kotlin_configurations_and_plugins():
    groovy = """plugins {
    id 'java'
    id "org.springframework.boot" version "3.2.0"
}
dependencies {
    implementation 'org.springframework.boot:spring-boot-starter-web:3.2.0'
    api group: 'com.google.guava', name: 'guava', version: '33.0'  // not a coordinate string
    runtimeOnly "org.postgresql:postgresql"
    testImplementation 'org.junit.jupiter:junit-jupiter:5.10' // tests
    annotationProcessor 'org.projectlombok:lombok:1.18'
    implementation project(':lib')
}
"""
    claims = manifest_claims(groovy, "app/build.gradle")
    assert said(claims, DEPENDS_ON) == [("app/build.gradle", DEPENDS_ON, "org.springframework.boot:spring-boot-starter-web"),
                                        ("app/build.gradle", DEPENDS_ON, "org.postgresql:postgresql")]
    assert said(claims, DEVELOPS_WITH) == [("app/build.gradle", DEVELOPS_WITH, "java"),
                                           ("app/build.gradle", DEVELOPS_WITH, "org.springframework.boot"),
                                           ("app/build.gradle", DEVELOPS_WITH, "org.junit.jupiter:junit-jupiter"),
                                           ("app/build.gradle", DEVELOPS_WITH, "org.projectlombok:lombok")], \
        "a map-style or project dependency is not guessed at"
    kotlin = 'dependencies {\n    implementation("io.ktor:ktor-server-core:2.3.0")\n    testImplementation(kotlin("test"))\n}\n'
    assert said(manifest_claims(kotlin, "build.gradle.kts")) == [("build.gradle.kts", DEPENDS_ON, "io.ktor:ktor-server-core")]


def test_a_gemfile_reads_groups_by_block_and_inline():
    content = """source 'https://rubygems.org'
gem 'rails', '~> 7.1'
gem "pg"
gem 'rspec-rails', group: :test
group :development, :test do
  gem 'rubocop'
  gem 'pry'
end
group :production do
  gem 'puma'
end
platforms :ruby do
  gem 'nokogiri'
end
# gem 'commented-out'
"""
    claims = manifest_claims(content, "Gemfile")
    assert said(claims, DEPENDS_ON) == [("Gemfile", DEPENDS_ON, "rails"), ("Gemfile", DEPENDS_ON, "pg"),
                                        ("Gemfile", DEPENDS_ON, "puma"), ("Gemfile", DEPENDS_ON, "nokogiri")]
    assert said(claims, DEVELOPS_WITH) == [("Gemfile", DEVELOPS_WITH, "rspec-rails"), ("Gemfile", DEVELOPS_WITH, "rubocop"),
                                           ("Gemfile", DEVELOPS_WITH, "pry")]
    assert {c.object: c.first_line for c in claims}["pry"] == 7


def test_a_composer_manifest_separates_require_from_require_dev_and_skips_the_platform():
    content = """{
  "name": "Acme/Shop",
  "require": {
    "php": ">=8.2",
    "ext-json": "*",
    "laravel/framework": "^11.0",
    "guzzlehttp/guzzle": "^7.8"
  },
  "require-dev": {
    "phpunit/phpunit": "^11"
  }
}
"""
    claims = manifest_claims(content, "composer.json")
    assert said(claims, DEFINES) == [("composer.json", DEFINES, "acme/shop")]
    assert said(claims, DEPENDS_ON) == [("acme/shop", DEPENDS_ON, "laravel/framework"), ("acme/shop", DEPENDS_ON, "guzzlehttp/guzzle")]
    assert said(claims, DEVELOPS_WITH) == [("acme/shop", DEVELOPS_WITH, "phpunit/phpunit")]
    assert {c.object: c.first_line for c in claims}["guzzlehttp/guzzle"] == 7


def test_a_dotnet_project_reads_package_references_and_marks_private_assets_as_build_time():
    content = """<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <AssemblyName>Acme.Api</AssemblyName>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
    <PackageReference Include="xunit" Version="2.6.1" />
    <PackageReference Include="StyleCop.Analyzers" Version="1.2.0" PrivateAssets="all" />
    <PackageReference Include="SourceGen">
      <PrivateAssets>all</PrivateAssets>
    </PackageReference>
    <ProjectReference Include="..\\Lib\\Lib.csproj" />
  </ItemGroup>
</Project>
"""
    claims = manifest_claims(content, "src/Api/Api.csproj")
    assert said(claims, DEFINES) == [("src/Api/Api.csproj", DEFINES, "acme.api")]
    assert said(claims, DEPENDS_ON) == [("acme.api", DEPENDS_ON, "newtonsoft.json"), ("acme.api", DEPENDS_ON, "xunit")]
    assert said(claims, DEVELOPS_WITH) == [("acme.api", DEVELOPS_WITH, "stylecop.analyzers"), ("acme.api", DEVELOPS_WITH, "sourcegen")]
    assert {c.object: c.first_line for c in claims}["sourcegen"] == 10, "a project reference names a file this cannot place"


def test_a_pipfile_reads_packages_and_dev_packages():
    content = """[[source]]
url = "https://pypi.org/simple"

[packages]
requests = "*"
Beautiful_Soup4 = {version = ">=4"}

[dev-packages]
pytest = "*"
"""
    claims = manifest_claims(content, "Pipfile")
    assert said(claims) == [("Pipfile", DEPENDS_ON, "requests"), ("Pipfile", DEPENDS_ON, "beautiful-soup4"),
                            ("Pipfile", DEVELOPS_WITH, "pytest")]
    assert {c.object: c.first_line for c in claims}["pytest"] == 9
