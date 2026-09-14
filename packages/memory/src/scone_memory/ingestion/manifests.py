"""What a project says it depends on, read from its package manifest.

A manifest is where a project writes down what it needs: ``pyproject.toml``,
``requirements.txt``, ``Pipfile``, ``package.json``, ``Cargo.toml``,
``go.mod``, ``pom.xml``, ``build.gradle(.kts)``, ``Gemfile``,
``composer.json`` and a .NET project file (``.csproj`` and kin). Left
unread, the graph knows every ``import requests`` and nothing about which
project declares requests, at what version, or only for its tests. Read
here, a manifest makes claims like a source file does -- quoted from the
line, cited to the file, extracted rather than stated -- and the questions
"what does this project depend on" and "who depends on this package" are
answered from the ledger like any other.

Two predicates, because they answer two questions:

- ``depends_on``: what the project needs to run, including its optional
  extras, which are still runtime;
- ``develops_with``: what it needs to build, test or document itself --
  build requirements, dependency groups, dev dependencies. A project does
  not run on pytest.

The object is the package's bare name, spelled as its index spells it (a
Python name lowercased with runs of ``-_.`` as one ``-``; a crate with
``_`` as ``-``; an npm, Composer or NuGet name lowercased; a Maven or
Gradle artifact as ``group:artifact``; a gem as named), with the extras,
version and marker left in the quote. So a package five manifests name
is one thing in the graph, and each manifest's line says what it asked
for. The subject is the project's declared name where it has one, else
the manifest's path. A Go ``// indirect`` requirement is left out: it is
what a dependency needs, not what the module declares; so is a Composer
platform requirement (``php``, ``ext-json``) and a .NET project
reference, which names a file this reader cannot place.

Names are not resolved against imports. The name a project depends on
and the name its code imports differ often enough (``beautifulsoup4``
and ``bs4``, ``Pillow`` and ``PIL``) that binding them would guess, and
the graph does not.
"""

from __future__ import annotations

import json
import re
import tomllib
from typing import Callable, Optional
import xml.etree.ElementTree as ElementTree

from .code import MAX_LINES, _line_starts
from .code_graph import DEFINES, MAX_CLAIMS, CodeClaim

#: What the project needs to run, extras included.
DEPENDS_ON = "depends_on"
#: What it needs to build, test or document itself.
DEVELOPS_WITH = "develops_with"

_NAMED = frozenset(("pyproject.toml", "package.json", "cargo.toml", "go.mod", "pom.xml", "build.gradle",
                    "build.gradle.kts", "gemfile", "composer.json", "pipfile"))
#: A .NET project file is known by its suffix; its name is the project's.
_PROJECT_SUFFIXES = (".csproj", ".fsproj", ".vbproj")
_PEP508_NAME = re.compile(r"\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_NORMALISE = re.compile(r"[-_.]+")


def is_manifest(path: str) -> bool:
    """Known by its name, wherever it sits: the named manifests, a .NET
    project file by its suffix, and any ``requirements*.txt`` or
    ``requirements/*.txt``."""
    if not path:
        return False
    parts = path.replace("\\", "/").lower().rsplit("/", 2)
    name = parts[-1]
    if name in _NAMED or name.endswith(_PROJECT_SUFFIXES):
        return True
    if not name.endswith(".txt"):
        return False
    return name.startswith("requirements") or (len(parts) > 1 and parts[-2] == "requirements")


def python_name(spec: str) -> Optional[str]:
    """The distribution a PEP 508 requirement names, normalised as the
    index normalises it, or None when the text does not start with one."""
    found = _PEP508_NAME.match(spec)
    return _NORMALISE.sub("-", found.group(1)).lower() if found else None


class _Claims:
    """Claims about one manifest, each quoting the line it was read on."""

    def __init__(self, content: str, path: str) -> None:
        self.path = path
        self.lines = content.split("\n")
        self.starts = _line_starts(content)
        self.content = content
        self.found: list[CodeClaim] = []

    def say(self, subject: str, predicate: str, obj: str, line: int) -> None:
        if len(self.found) >= MAX_CLAIMS:
            return
        line = max(1, line)
        text = self.lines[line - 1] if line - 1 < len(self.lines) else ""
        begins = len(self.content[: self.starts[line - 1]].encode()) if line - 1 < len(self.starts) else 0
        self.found.append(CodeClaim(subject, predicate, obj, text.strip(), line, begins, begins + len(text.encode())))

    def line_matching(self, pattern: "re.Pattern[str]", start: int = 0, stop: Optional[int] = None) -> int:
        """1-based number of the first line in ``[start, stop)`` the
        pattern matches, or 0."""
        for index in range(start, len(self.lines) if stop is None else min(stop, len(self.lines))):
            if pattern.search(self.lines[index]):
                return index + 1
        return 0


# --- TOML: where a table is, and which line a key or a value sits on ------

_HEADER = re.compile(r"^\s*\[\[?\s*(.+?)\s*\]\]?\s*(?:#.*)?$")


def _header_key(line: str) -> Optional[str]:
    found = _HEADER.match(line)
    if not found:
        return None
    return ".".join(part.strip().strip("'\"") for part in found.group(1).split("."))


class _Toml(_Claims):
    def __init__(self, content: str, path: str) -> None:
        super().__init__(content, path)
        self.headers = [(index, key) for index, key in
                        ((index, _header_key(line)) for index, line in enumerate(self.lines)) if key is not None]

    def table(self, *segments: str) -> Optional[tuple[int, int]]:
        """The 0-based line range a table's body occupies, header included."""
        wanted = ".".join(segments)
        for position, (index, key) in enumerate(self.headers):
            if key == wanted:
                stop = self.headers[position + 1][0] if position + 1 < len(self.headers) else len(self.lines)
                return index, stop
        return None

    def key_line(self, segments: tuple[str, ...], key: str) -> int:
        where = self.table(*segments)
        if where is None:
            return 0
        return self.line_matching(re.compile(r"^\s*['\"]?" + re.escape(key) + r"['\"]?\s*="), *where)

    def value_line(self, segments: tuple[str, ...], key: str, value: str) -> int:
        """The line inside a key's array where a string value is written,
        else the key's own line."""
        where = self.table(*segments)
        at = self.key_line(segments, key)
        if where is None or not at:
            return at
        quoted = re.compile("[\"']" + re.escape(value) + "[\"']")
        return self.line_matching(quoted, at - 1, where[1]) or at


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _table(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _pyproject(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return ()
    text = _Toml(content, path)
    project, poetry = _table(data.get("project")), _table(_table(_table(data.get("tool")).get("poetry")))
    subject = path
    name = project.get("name") if isinstance(project.get("name"), str) else poetry.get("name")
    if isinstance(name, str) and python_name(name):
        subject = python_name(name) or path
        at = text.key_line(("project",), "name") or text.key_line(("tool", "poetry"), "name")
        text.say(path, DEFINES, subject, at)

    def specs(predicate: str, segments: tuple[str, ...], key: str, value: object) -> None:
        for spec in _strings(value):
            package = python_name(spec)
            if package:
                text.say(subject, predicate, package, text.value_line(segments, key, spec))

    def keys(predicate: str, segments: tuple[str, ...], value: object) -> None:
        for key in _table(value):
            package = python_name(key)
            if package and package != "python":
                text.say(subject, predicate, package, text.key_line(segments, key))

    specs(DEVELOPS_WITH, ("build-system",), "requires", _table(data.get("build-system")).get("requires"))
    specs(DEPENDS_ON, ("project",), "dependencies", project.get("dependencies"))
    for group, listed in _table(project.get("optional-dependencies")).items():
        specs(DEPENDS_ON, ("project", "optional-dependencies"), group, listed)
    for group, listed in _table(data.get("dependency-groups")).items():
        specs(DEVELOPS_WITH, ("dependency-groups",), group, listed)
    keys(DEPENDS_ON, ("tool", "poetry", "dependencies"), poetry.get("dependencies"))
    keys(DEVELOPS_WITH, ("tool", "poetry", "dev-dependencies"), poetry.get("dev-dependencies"))
    for group, body in _table(poetry.get("group")).items():
        keys(DEVELOPS_WITH, ("tool", "poetry", "group", group, "dependencies"), _table(body).get("dependencies"))
    return tuple(text.found)


def _requirements(content: str, path: str) -> tuple[CodeClaim, ...]:
    text = _Claims(content, path)
    for number, line in enumerate(text.lines, start=1):
        bare = line.split(" #", 1)[0].strip()
        if not bare or bare.startswith(("#", "-")):
            continue
        package = python_name(bare)
        if package:
            text.say(path, DEPENDS_ON, package, number)
    return tuple(text.found)


def _package_json(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, dict):
        return ()
    text = _Claims(content, path)
    subject = path
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        subject = name.strip().lower()
        text.say(path, DEFINES, subject, text.line_matching(re.compile(r'^\s*"name"\s*:')))
    for section, predicate in (("dependencies", DEPENDS_ON), ("peerDependencies", DEPENDS_ON),
                               ("optionalDependencies", DEPENDS_ON), ("devDependencies", DEVELOPS_WITH)):
        listed = _table(data.get(section))
        if not listed:
            continue
        start = text.line_matching(re.compile(r'^\s*"' + section + r'"\s*:'))
        for package in listed:
            if isinstance(package, str) and package.strip():
                at = text.line_matching(re.compile(r'^\s*"' + re.escape(package) + r'"\s*:'), max(start - 1, 0))
                text.say(subject, predicate, package.strip().lower(), at or start)
    return tuple(text.found)


def _crate(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _cargo(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return ()
    text = _Toml(content, path)
    subject = path
    named = _table(data.get("package")).get("name")
    if isinstance(named, str) and named.strip():
        subject = _crate(named)
        text.say(path, DEFINES, subject, text.key_line(("package",), "name"))

    def crates(predicate: str, segments: tuple[str, ...], value: object) -> None:
        for key, spec in _table(value).items():
            if isinstance(spec, dict):
                real = spec.get("package")
                named = real if isinstance(real, str) and real.strip() else key
            elif isinstance(spec, str):
                named = key
            else:
                continue
            text.say(subject, predicate, _crate(named), text.key_line(segments, key))

    crates(DEPENDS_ON, ("dependencies",), data.get("dependencies"))
    crates(DEPENDS_ON, ("workspace", "dependencies"), _table(data.get("workspace")).get("dependencies"))
    targets = _table(data.get("target"))
    for target, body in targets.items():
        crates(DEPENDS_ON, ("target", target, "dependencies"), _table(body).get("dependencies"))
    crates(DEVELOPS_WITH, ("dev-dependencies",), data.get("dev-dependencies"))
    for target, body in targets.items():
        crates(DEVELOPS_WITH, ("target", target, "dev-dependencies"), _table(body).get("dev-dependencies"))
    crates(DEVELOPS_WITH, ("build-dependencies",), data.get("build-dependencies"))
    for target, body in targets.items():
        crates(DEVELOPS_WITH, ("target", target, "build-dependencies"), _table(body).get("build-dependencies"))
    return tuple(text.found)


_GO_MODULE = re.compile(r"^\s*module\s+(\S+)")
_GO_REQUIRE = re.compile(r"^\s*require\s+(\S+)\s+\S+")
_GO_ENTRY = re.compile(r"^\s*(\S+)\s+\S+")


def _go_mod(content: str, path: str) -> tuple[CodeClaim, ...]:
    text = _Claims(content, path)
    subject = path
    block = False
    for number, line in enumerate(text.lines, start=1):
        code, _, comment = line.partition("//")
        if block:
            if code.strip() == ")":
                block = False
                continue
            entry = _GO_ENTRY.match(code)
            if entry and "indirect" not in comment:
                text.say(subject, DEPENDS_ON, entry.group(1), number)
            continue
        module = _GO_MODULE.match(code)
        if module and subject == path:
            subject = module.group(1)
            text.say(path, DEFINES, subject, number)
            continue
        if re.match(r"^\s*require\s*\(\s*$", code):
            block = True
            continue
        single = _GO_REQUIRE.match(code)
        if single and "indirect" not in comment:
            text.say(subject, DEPENDS_ON, single.group(1), number)
    return tuple(text.found)


# --- Maven -----------------------------------------------------------------

def _local(tag: str) -> str:
    """An element's name without its namespace: a POM declares one."""
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ElementTree.Element, name: str) -> Optional[str]:
    for child in element:
        if _local(child.tag) == name and child.text and child.text.strip():
            return child.text.strip()
    return None


def _xml(content: str) -> Optional[ElementTree.Element]:
    try:
        return ElementTree.fromstring(content)
    except ElementTree.ParseError:
        return None


def _pom(content: str, path: str) -> tuple[CodeClaim, ...]:
    root = _xml(content)
    if root is None or _local(root.tag) != "project":
        return ()
    text = _Claims(content, path)
    parent = next((child for child in root if _local(child.tag) == "parent"), None)
    group = _child_text(root, "groupId") or (_child_text(parent, "groupId") if parent is not None else None)
    artifact = _child_text(root, "artifactId")
    subject = path
    if group and artifact:
        subject = f"{group}:{artifact}".lower()
        text.say(path, DEFINES, subject, text.line_matching(re.compile(r"<artifactId>\s*" + re.escape(artifact))))
    cursor = 0

    def coordinates(element: ElementTree.Element, predicate: str) -> None:
        nonlocal cursor
        group, artifact = _child_text(element, "groupId"), _child_text(element, "artifactId")
        if not group or not artifact:
            return
        at = text.line_matching(re.compile(r"<artifactId>\s*" + re.escape(artifact) + r"\s*<"), cursor)
        cursor = max(cursor, at)
        text.say(subject, predicate, f"{group}:{artifact}".lower(), at)

    for section in root:
        name = _local(section.tag)
        if name == "dependencies":
            for dependency in section:
                if _local(dependency.tag) == "dependency":
                    # A test-scoped dependency is what the project tests
                    # with, not what it runs on.
                    scope = (_child_text(dependency, "scope") or "compile").lower()
                    coordinates(dependency, DEVELOPS_WITH if scope == "test" else DEPENDS_ON)
        elif name == "build":
            for plugins in section:
                if _local(plugins.tag) == "plugins":
                    for plugin in plugins:
                        if _local(plugin.tag) == "plugin":
                            coordinates(plugin, DEVELOPS_WITH)
    return tuple(text.found)


# --- Gradle ----------------------------------------------------------------

#: A dependency line, Groovy or Kotlin: `implementation 'g:a:v'`,
#: `testImplementation("g:a:v")`. The configuration says what it is for.
_GRADLE = re.compile(r"^\s*(implementation|api|compileOnly|compileOnlyApi|runtimeOnly|testImplementation|"
                     r"testCompileOnly|testRuntimeOnly|annotationProcessor|kapt|ksp|classpath)\s*\(?\s*"
                     r"(?:group\s*[:=]\s*)?['\"]([^'\"]+)['\"]")
_GRADLE_DEVELOPS = frozenset(("testImplementation", "testCompileOnly", "testRuntimeOnly", "annotationProcessor",
                              "kapt", "ksp", "classpath"))
_GRADLE_PLUGIN = re.compile(r"^\s*id\s*\(?\s*['\"]([^'\"]+)['\"]")


def _gradle(content: str, path: str) -> tuple[CodeClaim, ...]:
    text = _Claims(content, path)
    for number, line in enumerate(text.lines, start=1):
        code = line.split("//", 1)[0]
        found = _GRADLE.match(code)
        if found:
            parts = found.group(2).split(":")
            if len(parts) >= 2 and parts[0] and parts[1]:
                predicate = DEVELOPS_WITH if found.group(1) in _GRADLE_DEVELOPS else DEPENDS_ON
                text.say(path, predicate, f"{parts[0]}:{parts[1]}".lower(), number)
            continue
        plugin = _GRADLE_PLUGIN.match(code)
        if plugin:
            text.say(path, DEVELOPS_WITH, plugin.group(1).lower(), number)
    return tuple(text.found)


# --- Bundler ---------------------------------------------------------------

_GEM = re.compile(r"^\s*gem\s+['\"]([^'\"]+)['\"](.*)$")
_GEM_GROUP = re.compile(r"^\s*group\s+(.+?)\s+do\b")
_DEVELOPMENT = re.compile(r":(?:test|development|ci|lint)\b")


def _gemfile(content: str, path: str) -> tuple[CodeClaim, ...]:
    text = _Claims(content, path)
    grouped: list[bool] = []  # for each open block, whether it is a development group
    for number, line in enumerate(text.lines, start=1):
        code = line.split("#", 1)[0]
        opened = _GEM_GROUP.match(code)
        if opened:
            grouped.append(bool(_DEVELOPMENT.search(opened.group(1))))
            continue
        if re.match(r"^\s*\w[\w.]*.*\bdo\b\s*(?:\|[^|]*\|)?\s*$", code) and not opened:
            grouped.append(False)  # some other block: `source ... do`, `platforms ... do`
            continue
        if code.strip() == "end":
            if grouped:
                grouped.pop()
            continue
        gem = _GEM.match(code)
        if gem:
            inline = _DEVELOPMENT.search(gem.group(2)) if "group" in gem.group(2) else None
            development = any(grouped) or inline is not None
            text.say(path, DEVELOPS_WITH if development else DEPENDS_ON, gem.group(1).lower(), number)
    return tuple(text.found)


# --- Composer --------------------------------------------------------------

def _composer(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return ()
    if not isinstance(data, dict):
        return ()
    text = _Claims(content, path)
    subject = path
    name = data.get("name")
    if isinstance(name, str) and name.strip():
        subject = name.strip().lower()
        text.say(path, DEFINES, subject, text.line_matching(re.compile(r'^\s*"name"\s*:')))
    for section, predicate in (("require", DEPENDS_ON), ("require-dev", DEVELOPS_WITH)):
        listed = _table(data.get(section))
        if not listed:
            continue
        start = text.line_matching(re.compile(r'^\s*"' + section + r'"\s*:'))
        for package in listed:
            # `php` and `ext-*`/`lib-*` are the platform, not packages.
            if not isinstance(package, str) or "/" not in package:
                continue
            at = text.line_matching(re.compile(r'^\s*"' + re.escape(package) + r'"\s*:'), max(start - 1, 0))
            text.say(subject, predicate, package.strip().lower(), at or start)
    return tuple(text.found)


# --- .NET ------------------------------------------------------------------

def _dotnet(content: str, path: str) -> tuple[CodeClaim, ...]:
    root = _xml(content)
    if root is None or _local(root.tag) != "Project":
        return ()
    text = _Claims(content, path)
    subject = path
    for group in root:
        if _local(group.tag) != "PropertyGroup":
            continue
        named = _child_text(group, "PackageId") or _child_text(group, "AssemblyName")
        if named:
            subject = named.lower()
            text.say(path, DEFINES, subject, text.line_matching(re.compile(r"<(?:PackageId|AssemblyName)>")))
            break
    cursor = 0
    for group in root:
        if _local(group.tag) != "ItemGroup":
            continue
        for item in group:
            if _local(item.tag) != "PackageReference":
                continue
            package = item.get("Include") or item.get("Update")
            if not package:
                continue
            # PrivateAssets="all" is a build-time package: an analyzer,
            # a source generator, a test SDK.
            private = (item.get("PrivateAssets") or _child_text(item, "PrivateAssets") or "").lower()
            at = text.line_matching(re.compile(r'<PackageReference\s[^>]*(?:Include|Update)="' + re.escape(package) + '"'), cursor)
            cursor = max(cursor, at)
            text.say(subject, DEVELOPS_WITH if private == "all" else DEPENDS_ON, package.lower(), at)
    return tuple(text.found)


# --- Pipfile ---------------------------------------------------------------

def _pipfile(content: str, path: str) -> tuple[CodeClaim, ...]:
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError:
        return ()
    text = _Toml(content, path)
    for section, predicate in (("packages", DEPENDS_ON), ("dev-packages", DEVELOPS_WITH)):
        for key in _table(data.get(section)):
            package = python_name(key)
            if package:
                text.say(path, predicate, package, text.key_line((section,), key))
    return tuple(text.found)


def _reader(path: str) -> Optional[Callable[[str, str], tuple[CodeClaim, ...]]]:
    name = path.replace("\\", "/").lower().rsplit("/", 1)[-1]
    if name == "pyproject.toml":
        return _pyproject
    if name == "package.json":
        return _package_json
    if name == "cargo.toml":
        return _cargo
    if name == "go.mod":
        return _go_mod
    if name == "pom.xml":
        return _pom
    if name in ("build.gradle", "build.gradle.kts"):
        return _gradle
    if name == "gemfile":
        return _gemfile
    if name == "composer.json":
        return _composer
    if name == "pipfile":
        return _pipfile
    if name.endswith(_PROJECT_SUFFIXES):
        return _dotnet
    return _requirements if is_manifest(path) else None


def manifest_claims(content: str, path: str) -> tuple[CodeClaim, ...]:
    """What a manifest declares and depends on, as claims about it.
    Empty for a file that is not a manifest, does not parse, or is
    longer than a source file is read."""
    if not content or content.count("\n") > MAX_LINES:
        return ()
    read = _reader(path)
    return read(content, path) if read is not None else ()
