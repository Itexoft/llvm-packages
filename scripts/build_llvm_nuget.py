#!/usr/bin/env python3
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_SUFFIXES = (".tar.xz", ".tar.gz", ".zip")
GITHUB_REPO = "llvm/llvm-project"
GITHUB_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
GITHUB_RELEASE_TAG_URL = (
    f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{{tag}}"
)
GITHUB_RELEASES_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases"
PACKAGE_PREFIX = "LLVM"
SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
ELF_MAGIC = b"\x7fELF"
MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",
    b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca",
    b"\xca\xfe\xba\xbf",
    b"\xbf\xba\xfe\xca",
}
PE_MAGIC = b"MZ"


def log(message):
    print(message, flush=True)


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_package_json(path):
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        fail(f"Missing {path}")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path}: {exc}")
    if not isinstance(data, dict):
        fail("packages.json must be a JSON object of component -> spec")
    normalized = {}
    for key, value in data.items():
        if not isinstance(key, str):
            fail("packages.json keys must be strings")
        if not isinstance(value, dict):
            fail(f"packages.json value for {key} must be an object")
        if "bin" not in value or "merge" not in value or "version" not in value:
            fail(
                f"packages.json value for {key} must include bin, merge, and version"
            )
        patterns = value.get("bin")
        merge = value.get("merge")
        version = value.get("version")
        if not isinstance(patterns, list) or not all(
            isinstance(item, str) for item in patterns
        ):
            fail(f"packages.json bin for {key} must be an array of strings")
        if not patterns:
            fail(f"packages.json bin for {key} must not be empty")
        if not isinstance(merge, bool):
            fail(f"packages.json merge for {key} must be true or false")
        if not isinstance(version, str) or not version.strip():
            fail(f"packages.json version for {key} must be a string")
        normalized[key] = {"bin": patterns, "merge": merge, "version": version.strip()}
    return normalized


def github_headers():
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "llvm-nuget-packager",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def fetch_latest_release():
    req = urllib.request.Request(GITHUB_RELEASE_URL, headers=github_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        fail(f"Failed to fetch latest release: {exc}")
    tag_name = data.get("tag_name")
    if not tag_name:
        fail("Missing tag_name in GitHub release response")
    assets = data.get("assets") or []
    normalized = []
    for asset in assets:
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if name and url:
            normalized.append({"name": name, "url": url})
    return tag_name, normalized


def fetch_release(tag_name):
    url = GITHUB_RELEASE_TAG_URL.format(tag=tag_name)
    req = urllib.request.Request(url, headers=github_headers())
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        fail(f"Failed to fetch release {tag_name}: {exc}")
    assets = data.get("assets") or []
    normalized = []
    for asset in assets:
        name = asset.get("name")
        url = asset.get("browser_download_url")
        if name and url:
            normalized.append({"name": name, "url": url})
    return tag_name, normalized


def fetch_release_index():
    releases = []
    page = 1
    while True:
        url = f"{GITHUB_RELEASES_URL}?per_page=100&page={page}"
        req = urllib.request.Request(url, headers=github_headers())
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            fail(f"Failed to fetch release list: {exc}")
        if not isinstance(data, list) or not data:
            break
        for entry in data:
            if entry.get("draft") or entry.get("prerelease"):
                continue
            tag = entry.get("tag_name")
            version = try_version_from_tag(tag)
            if not version:
                continue
            releases.append({"tag": tag, "version": version})
        if len(data) < 100:
            break
        page += 1
    return releases


def resolve_version_pattern(pattern, releases):
    matches = []
    for entry in releases:
        version = entry["version"]
        if version_matches_pattern(version, pattern):
            matches.append(entry)
    if not matches:
        fail(f"No releases match version pattern {pattern}")
    matches.sort(key=lambda item: semver_key(item["version"]))
    return matches[-1]


def normalize_release_tag(value):
    if not value:
        return None
    tag = value.strip()
    if not tag:
        return None
    if tag.startswith("llvmorg-"):
        return tag
    return f"llvmorg-{tag}"


def resolve_component_release(version_spec, releases):
    if "*" in version_spec:
        if releases is None:
            releases = fetch_release_index()
        match = resolve_version_pattern(version_spec, releases)
        return match["tag"], match["version"]

    if version_spec.startswith("llvmorg-"):
        tag = version_spec
        return tag, version_from_tag(tag)

    if not SEMVER_RE.match(version_spec):
        fail(f"Invalid version {version_spec} in packages.json")
    return normalize_release_tag(version_spec), version_spec


def version_from_tag(tag_name):
    prefix = "llvmorg-"
    if not tag_name.startswith(prefix):
        fail(f"Release tag {tag_name} does not start with {prefix}")
    version = tag_name[len(prefix) :]
    if not SEMVER_RE.match(version):
        fail(f"Release tag {tag_name} produces invalid NuGet version {version}")
    return version


def try_version_from_tag(tag_name):
    prefix = "llvmorg-"
    if not tag_name or not tag_name.startswith(prefix):
        return None
    version = tag_name[len(prefix) :]
    if not SEMVER_RE.match(version):
        return None
    return version


def semver_key(version):
    core = version.split("-", 1)[0].split("+", 1)[0]
    parts = core.split(".")
    if len(parts) != 3:
        return None
    try:
        major, minor, patch = (int(part) for part in parts)
    except ValueError:
        return None
    is_release = "-" not in version
    prerelease = ""
    if not is_release:
        prerelease = version.split("-", 1)[1].split("+", 1)[0]
    return (major, minor, patch, is_release, prerelease)


def version_matches_pattern(version, pattern):
    if "*" not in pattern:
        return version == pattern
    regex = "^" + re.escape(pattern).replace("\\*", ".*") + "$"
    return re.match(regex, version) is not None


def is_archive(name):
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def is_target_asset(name):
    lower = name.lower()
    if "llvm" not in lower:
        return False
    return "src" not in lower


def determine_rid(name):
    lower = name.lower()
    os_part = None
    if any(token in lower for token in ["macos", "osx", "darwin"]):
        os_part = "osx"
    elif "linux" in lower:
        os_part = "linux"
    elif any(token in lower for token in ["windows", "win", "msvc"]):
        os_part = "win"

    arch_part = None
    if any(token in lower for token in ["aarch64", "arm64"]):
        arch_part = "arm64"
    elif any(token in lower for token in ["x86_64", "amd64", "x64"]):
        arch_part = "x64"

    if os_part and arch_part:
        return f"{os_part}-{arch_part}"
    return None


def collect_release_rids(assets):
    rids = set()
    for asset in assets:
        name = asset.get("name")
        if not name:
            continue
        if not is_archive(name):
            continue
        if not is_target_asset(name):
            continue
        rid = determine_rid(name)
        if rid:
            rids.add(rid)
    return sorted(rids)


def package_id_for(component_key, rid=None):
    if rid:
        return f"{PACKAGE_PREFIX}.{component_key}.{rid}".lower()
    return f"{PACKAGE_PREFIX}.{component_key}".lower()


def download_asset(url, destination):
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "llvm-nuget-packager",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(destination, "wb") as handle:
        shutil.copyfileobj(resp, handle)


def nuget_version_exists(package_id, version, cache):
    key = (package_id.lower(), version.lower())
    if key in cache:
        return cache[key]
    url = f"https://api.nuget.org/v3-flatcontainer/{key[0]}/index.json"
    req = urllib.request.Request(url, headers={"User-Agent": "llvm-nuget-packager"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError:
        cache[key] = False
        return False
    except OSError:
        log(f"Warning: failed to query NuGet for {package_id}")
        cache[key] = False
        return False
    versions = data.get("versions") or []
    exists = any(str(item).lower() == key[1] for item in versions)
    cache[key] = exists
    return exists


def extract_archive(archive_path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(destination)
    else:
        with tarfile.open(archive_path, "r:*") as archive:
            archive.extractall(destination)


def collect_files(root_dir):
    files = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root_dir).as_posix()
        rel_path = PurePosixPath(rel)
        if rel_path.suffix:
            rel_no_ext = rel_path.with_suffix("").as_posix()
        else:
            rel_no_ext = rel_path.as_posix()
        files.append({"abs": path, "rel": rel, "rel_no_ext": rel_no_ext})
    return files


def normalize_rel_path(value):
    value = value.replace("\\", "/").lstrip("/")
    parts = []
    for part in value.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def rel_without_extension(rel):
    rel_path = PurePosixPath(rel)
    if rel_path.suffix:
        return rel_path.with_suffix("").as_posix()
    return rel_path.as_posix()


def resolve_link_target(rel, linkname):
    if not linkname:
        return None
    target = PurePosixPath(rel).parent / linkname
    normalized = normalize_rel_path(target.as_posix())
    return normalized


class ArchiveIndex:
    def __init__(self, archive_path):
        self.archive_path = Path(archive_path)
        self.entries = []
        self.meta_by_rel = {}
        self.kind = "zip" if self.archive_path.name.lower().endswith(".zip") else "tar"
        self._build()

    def _add_entry(self, rel, member_name, mode=None, is_link=False, linkname=None):
        rel_no_ext = rel_without_extension(rel)
        entry = {"rel": rel, "rel_no_ext": rel_no_ext, "member": rel}
        self.entries.append(entry)
        self.meta_by_rel[rel] = {
            "member_name": member_name,
            "mode": mode,
            "is_link": is_link,
            "linkname": linkname,
        }

    def _build(self):
        if self.kind == "zip":
            with zipfile.ZipFile(self.archive_path) as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    rel = normalize_rel_path(info.filename)
                    if rel is None:
                        continue
                    mode = (info.external_attr >> 16) & 0o777
                    self._add_entry(rel, info.filename, mode=mode)
        else:
            with tarfile.open(self.archive_path, "r:*") as archive:
                for member in archive.getmembers():
                    if member.isdir():
                        continue
                    if not (member.isfile() or member.issym() or member.islnk()):
                        continue
                    rel = normalize_rel_path(member.name)
                    if rel is None:
                        continue
                    self._add_entry(
                        rel,
                        member.name,
                        mode=member.mode,
                        is_link=member.issym() or member.islnk(),
                        linkname=member.linkname,
                    )

    def resolve_real_rel(self, rel):
        current = rel
        seen = set()
        while True:
            meta = self.meta_by_rel.get(current)
            if not meta:
                return None
            if not meta["is_link"]:
                return current
            if current in seen:
                return None
            seen.add(current)
            target_rel = resolve_link_target(current, meta["linkname"])
            if not target_rel:
                return None
            current = target_rel

    def extract(self, rel, dest_root):
        dest_path = Path(dest_root) / rel
        return self.extract_to(rel, dest_path)

    def extract_to(self, rel, dest_path):
        if rel not in self.meta_by_rel:
            return None
        dest_path = Path(dest_path)
        if dest_path.exists():
            return dest_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        real_rel = self.resolve_real_rel(rel)
        if not real_rel:
            return None
        meta = self.meta_by_rel[real_rel]
        if self.kind == "zip":
            with zipfile.ZipFile(self.archive_path) as archive:
                with archive.open(meta["member_name"]) as src, open(
                    dest_path, "wb"
                ) as dst:
                    shutil.copyfileobj(src, dst)
        else:
            with tarfile.open(self.archive_path, "r:*") as archive:
                member = archive.getmember(meta["member_name"])
                handle = archive.extractfile(member)
                if handle is None:
                    return None
                with handle as src, open(dest_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        if meta["mode"]:
            try:
                os.chmod(dest_path, meta["mode"])
            except OSError:
                pass
        return dest_path


def glob_match(path, pattern):
    path = path.lower()
    pattern = pattern.lower()
    parts = [part for part in path.split("/") if part]
    pattern_parts = [part for part in pattern.split("/") if part]

    collapsed = []
    for part in pattern_parts:
        if part == "**" and collapsed and collapsed[-1] == "**":
            continue
        collapsed.append(part)
    pattern_parts = collapsed

    def match_parts(path_parts, pat_parts):
        if not pat_parts:
            return not path_parts
        head = pat_parts[0]
        if head == "**":
            if len(pat_parts) == 1:
                return True
            for idx in range(len(path_parts) + 1):
                if match_parts(path_parts[idx:], pat_parts[1:]):
                    return True
            return False
        if not path_parts:
            return False
        if not fnmatch.fnmatchcase(path_parts[0], head):
            return False
        return match_parts(path_parts[1:], pat_parts[1:])

    return match_parts(parts, pattern_parts)


def find_component_files(patterns, files):
    selected = []
    missing = []
    counts = {}
    for pattern in patterns:
        pattern_norm = pattern.replace("\\", "/")
        matches = [
            item
            for item in files
            if glob_match(item["rel_no_ext"], pattern_norm)
        ]
        matches.sort(key=lambda item: item["rel"])
        counts[pattern_norm] = len(matches)
        if not matches:
            missing.append(pattern_norm)
            continue
        selected.extend(matches)
    if not selected:
        return None, missing, counts
    unique = {}
    for item in selected:
        unique[item["rel"]] = item
    return list(unique.values()), missing, counts


def detect_file_type(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(4)
    except OSError:
        return None
    if head.startswith(ELF_MAGIC):
        return "elf"
    if head[:2] == PE_MAGIC:
        return "pe"
    if head in MACHO_MAGICS:
        return "macho"
    return None


def parse_elf_dynamic(path, readelf_path, llvm_readobj_path):
    command = None
    if readelf_path:
        command = [readelf_path, "-d", str(path)]
    elif llvm_readobj_path:
        command = [llvm_readobj_path, "--dynamic-table", str(path)]
    else:
        return [], []
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        )
    except subprocess.CalledProcessError:
        return [], []
    runpaths = []
    needed = []
    for line in output.splitlines():
        if "(RUNPATH)" in line or "(RPATH)" in line:
            match = re.search(r"\\[(.*)\\]", line)
            if match:
                runpaths.append(match.group(1))
        elif "(NEEDED)" in line:
            match = re.search(r"\\[(.*)\\]", line)
            if match:
                needed.append(match.group(1))
    return runpaths, needed


def parse_macho_info(path, llvm_objdump_path):
    if not llvm_objdump_path:
        return [], []
    rpaths = []
    dylibs = []
    try:
        output = subprocess.check_output(
            [llvm_objdump_path, "--macho", "--rpaths", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        output = ""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        rpaths.append(stripped)
    try:
        output = subprocess.check_output(
            [llvm_objdump_path, "--macho", "--dylibs-used", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        output = ""
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        dylibs.append(stripped.split()[0])
    return rpaths, dylibs


def parse_pe_imports(path, llvm_readobj_path):
    if not llvm_readobj_path:
        return []
    try:
        output = subprocess.check_output(
            [llvm_readobj_path, "--coff-imports", str(path)],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return []
    imports = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Name:"):
            name = line.split("Name:", 1)[1].strip()
            if name:
                imports.append(name)
    return imports


def select_lib_entries(names, lib_index):
    deps = []
    for name in names:
        matches = lib_index.get(name.lower())
        if matches:
            deps.append(matches[0])
    return deps


def select_elf_dependencies(path, lib_index, readelf_path, llvm_readobj_path):
    _, needed = parse_elf_dynamic(path, readelf_path, llvm_readobj_path)
    if not needed:
        return []
    return select_lib_entries(needed, lib_index)


def select_macho_dependencies(path, lib_index, llvm_objdump_path):
    _, dylibs = parse_macho_info(path, llvm_objdump_path)
    deps = []
    for dep in dylibs:
        if dep.startswith("/"):
            continue
        name = PurePosixPath(dep).name
        matches = lib_index.get(name.lower())
        if matches:
            deps.append(matches[0])
    return deps


def select_pe_dependencies(path, lib_index, llvm_readobj_path):
    imports = parse_pe_imports(path, llvm_readobj_path)
    if not imports:
        return []
    return select_lib_entries(imports, lib_index)


def build_lib_index(entries, lib_prefix):
    prefix = lib_prefix.rstrip("/") + "/"
    index = {}
    for entry in entries:
        rel = entry["rel"]
        if not rel.startswith(prefix):
            continue
        name = PurePosixPath(rel).name.lower()
        index.setdefault(name, []).append(entry)
    for name in index:
        index[name].sort(key=lambda item: item["rel"])
    return index


def build_readme_map(repo_root):
    readme_map = {}
    for path in repo_root.glob("README.*.md"):
        name = path.name
        if not name.startswith("README.") or not name.lower().endswith(".md"):
            continue
        component = name[len("README.") : -len(".md")]
        if not component:
            continue
        readme_map[component.lower()] = path
    return readme_map


def output_rel_for_entry(entry):
    rel = entry["rel"]
    root_prefix = root_prefix_for_rel(rel)
    if root_prefix:
        prefix = root_prefix + "/"
        if rel.startswith(prefix):
            return rel[len(prefix) :]
    return rel


def ensure_extracted(entry, archive_index, content_dir, extracted, output_rel):
    if output_rel in extracted:
        return extracted[output_rel]
    dest_path = Path(content_dir) / output_rel
    dest = archive_index.extract_to(entry["rel"], dest_path)
    if dest is None:
        return None
    extracted[output_rel] = dest
    return dest


def resolve_dependencies(entry_paths, archive_index, lib_index, content_dir, tools, extracted):
    if not lib_index:
        return []
    resolved = {}
    queue = [Path(path) for path in entry_paths]
    visited = set()
    while queue:
        current = queue.pop()
        try:
            current_key = current.resolve()
        except OSError:
            continue
        if current_key in visited:
            continue
        visited.add(current_key)
        file_type = detect_file_type(current)
        if file_type == "elf":
            deps = select_elf_dependencies(
                current, lib_index, tools["readelf"], tools["llvm_readobj"]
            )
        elif file_type == "macho":
            deps = select_macho_dependencies(current, lib_index, tools["llvm_objdump"])
        elif file_type == "pe":
            deps = select_pe_dependencies(
                current, lib_index, tools["llvm_readobj"]
            )
        else:
            deps = []
        for entry in deps:
            output_rel = output_rel_for_entry(entry)
            dest = ensure_extracted(
                entry, archive_index, content_dir, extracted, output_rel
            )
            if dest is None:
                continue
            if output_rel in resolved:
                continue
            resolved[output_rel] = dest
            queue.append(dest)
    return list(resolved.values())


def root_prefix_for_rel(rel):
    parts = rel.split("/")
    for idx in range(len(parts) - 2, -1, -1):
        if parts[idx] == "bin":
            return "/".join(parts[:idx])
    for idx in range(len(parts) - 2, -1, -1):
        if parts[idx] == "lib":
            return "/".join(parts[:idx])
    return ""


def run_command(command):
    subprocess.run(command, check=True)


def log_nupkg_tree(nupkg_path):
    try:
        with zipfile.ZipFile(nupkg_path) as archive:
            names = [
                name.rstrip("/")
                for name in archive.namelist()
                if name and not name.endswith("/")
            ]
    except (OSError, zipfile.BadZipFile) as exc:
        log(f"NUPKG tree: failed to read {nupkg_path}: {exc}")
        return

    tree = {}
    for name in names:
        parts = [part for part in name.split("/") if part]
        node = tree
        for part in parts:
            node = node.setdefault(part, {})

    log(f"NUPKG tree: {Path(nupkg_path).name}")

    def render(node, prefix=""):
        items = sorted(node.items(), key=lambda item: item[0])
        for idx, (name, child) in enumerate(items):
            is_last = idx == len(items) - 1
            connector = "`-- " if is_last else "|-- "
            log(prefix + connector + name)
            extension = "    " if is_last else "|   "
            if child:
                render(child, prefix + extension)

    render(tree)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    packages_json_path = repo_root / "packages.json"
    template_path = repo_root / "llvm-package-template.csproj"

    package_map = load_package_json(packages_json_path)
    readme_map = build_readme_map(repo_root)
    if not template_path.exists():
        fail(f"Missing {template_path}")

    release_index = None
    for spec in package_map.values():
        if "*" in spec["version"]:
            release_index = fetch_release_index()
            break

    components = []
    for component, spec in package_map.items():
        tag_name, version = resolve_component_release(spec["version"], release_index)
        components.append(
            {
                "name": component,
                "patterns": spec["bin"],
                "merge": spec["merge"],
                "tag": tag_name,
                "version": version,
            }
        )

    release_groups = {}
    for component in components:
        tag = component["tag"]
        group = release_groups.setdefault(
            tag, {"version": component["version"], "components": []}
        )
        if group["version"] != component["version"]:
            fail(f"Release tag {tag} maps to multiple versions")
        group["components"].append(component)

    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rid_results = {}
    built_packages = set()
    built_package_versions = []
    skipped_existing = set()

    with tempfile.TemporaryDirectory() as temp_root:
        temp_root = Path(temp_root)
        work_dir = temp_root / "work"
        work_dir.mkdir(parents=True, exist_ok=True)
        project_path = work_dir / template_path.name
        shutil.copy2(template_path, project_path)
        placeholder_dir = work_dir / "placeholder"
        placeholder_dir.mkdir(parents=True, exist_ok=True)
        log("Template: restore")
        restore_command = [
            "dotnet",
            "restore",
            str(project_path),
            "--nologo",
            f"-p:PackageContentDir={placeholder_dir}",
        ]
        run_command(restore_command)

        tools = {
            "readelf": shutil.which("readelf"),
            "llvm_readobj": shutil.which("llvm-readobj"),
            "llvm_objdump": shutil.which("llvm-objdump"),
        }

        if not tools["readelf"] and not tools["llvm_readobj"]:
            log("Warning: readelf/llvm-readobj not found; ELF dependencies skipped.")
        if not tools["llvm_objdump"]:
            log("Warning: llvm-objdump not found; Mach-O dependencies skipped.")
        if not tools["llvm_readobj"]:
            log("Warning: llvm-readobj not found; PE dependencies skipped.")

        processed_any = False
        nuget_cache = {}
        for tag_name, group in sorted(release_groups.items()):
            version = group["version"]
            log(f"Release tag: {tag_name}")
            log(f"Package version: {version}")
            tag_name, assets = fetch_release(tag_name)
            rids = collect_release_rids(assets)
            if not rids:
                log(f"Release tag: {tag_name}: skip (no target assets)")
                skipped_existing.add(f"{tag_name}@{version}")
                continue

            package_exists_map = {}

            def package_exists(package_id):
                exists = package_exists_map.get(package_id)
                if exists is not None:
                    return exists
                exists = nuget_version_exists(package_id, version, nuget_cache)
                package_exists_map[package_id] = exists
                return exists

            expected_ids = set()
            for component in group["components"]:
                component_key = component["name"].lower()
                if component["merge"]:
                    expected_ids.add(package_id_for(component_key))
                else:
                    for rid in rids:
                        expected_ids.add(package_id_for(component_key, rid))

            all_existing = True
            for package_id in sorted(expected_ids):
                if not package_exists(package_id):
                    all_existing = False
                    break

            if all_existing:
                log(
                    f"Release tag: {tag_name}: skip (all packages already on NuGet)"
                )
                for package_id in expected_ids:
                    skipped_existing.add(f"{package_id}@{version}")
                continue

            merge_states = {}
            merge_needed = False
            for component in group["components"]:
                if not component["merge"]:
                    continue
                component_key = component["name"].lower()
                package_id = package_id_for(component_key)
                if package_exists(package_id):
                    log(f"COMPONENT {component_key}: skip (nuget has {version})")
                    skipped_existing.add(f"{package_id}@{version}")
                    component["skip"] = True
                    continue
                merge_needed = True
                merge_states[component_key] = {
                    "content_root": temp_root / f"content_{package_id}",
                    "rids": {},
                }

            rid_needed = None
            if not merge_needed:
                rid_needed = {}
                non_merge = [
                    component
                    for component in group["components"]
                    if not component["merge"]
                ]
                for rid in rids:
                    need = False
                    for component in non_merge:
                        component_key = component["name"].lower()
                        package_id = package_id_for(component_key, rid)
                        if not package_exists(package_id):
                            need = True
                            break
                    rid_needed[rid] = need

            for asset in assets:
                name = asset["name"]
                url = asset["url"]

                if not is_archive(name):
                    log(f"ASSET {name}: skip (not archive)")
                    continue

                if not is_target_asset(name):
                    log(f"ASSET {name}: skip (name prefix)")
                    continue

                rid = determine_rid(name)
                if not rid:
                    log(f"ASSET {name}: skip (rid not detected)")
                    continue

                if not merge_needed and rid_needed is not None:
                    if not rid_needed.get(rid, True):
                        log(
                            f"ASSET {name}: skip (nuget has {version} for all components)"
                        )
                        continue

                log(f"ASSET {name}: {rid}")

                asset_path = temp_root / name
                download_asset(url, asset_path)

                archive_index = ArchiveIndex(asset_path)
                files = archive_index.entries
                if not files:
                    log(f"ASSET {name}: no files in archive")
                    try:
                        asset_path.unlink()
                    except OSError:
                        pass
                    continue

                processed_any = True
                lib_index_cache = {}
                for component in group["components"]:
                    if component.get("skip"):
                        continue
                    component_key = component["name"].lower()
                    patterns = component["patterns"]
                    if not component["merge"]:
                        package_id = package_id_for(component_key, rid)
                        if package_id in built_packages:
                            continue
                        if package_exists(package_id):
                            log(
                                f"COMPONENT {component_key} ({rid}): skip (nuget has {version})"
                            )
                            built_packages.add(package_id)
                            skipped_existing.add(f"{package_id}@{version}")
                            continue

                    matched_files, missing_patterns, pattern_counts = find_component_files(
                        patterns, files
                    )
                    if not matched_files:
                        log(
                            f"COMPONENT {component_key} ({rid}): skip (no patterns matched)"
                        )
                        continue
                    matched_pattern_count = sum(
                        1 for count in pattern_counts.values() if count > 0
                    )
                    log(
                        f"COMPONENT {component_key} ({rid}): matched patterns {matched_pattern_count}/{len(patterns)}"
                    )
                    if missing_patterns:
                        log(
                            f"COMPONENT {component_key} ({rid}): missing {len(missing_patterns)} patterns"
                        )
                        for pattern in missing_patterns:
                            log(f"  missing: {pattern}")
                    multi_match = [
                        (pattern, count)
                        for pattern, count in pattern_counts.items()
                        if count > 1
                    ]
                    if multi_match:
                        log(f"COMPONENT {component_key} ({rid}): multi-match patterns")
                        for pattern, count in multi_match:
                            log(f"  matched: {pattern} -> {count}")

                    root_prefix = root_prefix_for_rel(matched_files[0]["rel"])
                    lib_prefix = f"{root_prefix}/lib" if root_prefix else "lib"
                    lib_index = lib_index_cache.get(lib_prefix)
                    if lib_index is None:
                        lib_index = build_lib_index(files, lib_prefix)
                        lib_index_cache[lib_prefix] = lib_index

                    if component["merge"]:
                        state = merge_states.get(component_key)
                        if state is None:
                            continue
                        rid_state = state["rids"].setdefault(
                            rid, {"extracted": {}}
                        )
                        content_dir = (
                            state["content_root"] / f"runtimes/{rid}/native"
                        )
                        extracted = rid_state["extracted"]
                    else:
                        content_dir = temp_root / f"content_{package_id}"
                        extracted = {}

                    matched_paths = []
                    for entry in matched_files:
                        output_rel = output_rel_for_entry(entry)
                        path = ensure_extracted(
                            entry, archive_index, content_dir, extracted, output_rel
                        )
                        if path:
                            matched_paths.append(path)

                    if not matched_paths:
                        continue

                    resolve_dependencies(
                        matched_paths,
                        archive_index,
                        lib_index,
                        content_dir,
                        tools,
                        extracted,
                    )
                    log(
                        f"COMPONENT {component_key} ({rid}): packaging {len(extracted)} files"
                    )

                    if component["merge"]:
                        if rid not in rid_results:
                            rid_results[rid] = {}
                        rid_results[rid][component_key] = sorted(extracted.keys())
                        continue

                    built_packages.add(package_id)

                    readme_path = readme_map.get(component_key)
                    readme_pack_path = None
                    if readme_path:
                        readme_pack_path = work_dir / "README.md"
                        shutil.copy2(readme_path, readme_pack_path)
                        log(
                            f"COMPONENT {component_key} ({rid}): readme {readme_path.name} -> README.md"
                        )

                    log(f"COMPONENT {component_key} ({rid}): pack")
                    pack_command = [
                        "dotnet",
                        "pack",
                        str(project_path),
                        "--nologo",
                        "--no-build",
                        "--no-restore",
                        f"-p:PackageId={package_id}",
                        f"-p:Version={version}",
                        f"-p:PackageRid={rid}",
                        f"-p:PackageComponent={component_key}",
                        f"-p:PackageContentDir={content_dir}",
                        f"-p:PackageOutputPath={artifacts_dir}",
                        "-p:NoWarn=NU5128",
                        "-p:IncludeSymbols=false",
                        "-p:IncludeSource=false",
                    ]
                    if readme_pack_path:
                        pack_command.extend(
                            [
                                "-p:PackageReadmeFile=README.md",
                                f"-p:PackageReadmePath={readme_pack_path}",
                            ]
                        )
                    run_command(pack_command)

                    nupkg_path = artifacts_dir / f"{package_id}.{version}.nupkg"
                    if not nupkg_path.exists():
                        fail(f"Expected package not found: {nupkg_path}")
                    log(f"COMPONENT {component_key} ({rid}): output {nupkg_path}")
                    log_nupkg_tree(nupkg_path)

                    if rid not in rid_results:
                        rid_results[rid] = {}
                    rid_results[rid][component_key] = sorted(extracted.keys())
                    built_package_versions.append(f"{package_id}@{version}")
                    shutil.rmtree(content_dir, ignore_errors=True)

                try:
                    asset_path.unlink()
                except OSError:
                    pass

            for component in group["components"]:
                if not component["merge"]:
                    continue
                if component.get("skip"):
                    continue
                component_key = component["name"].lower()
                state = merge_states.get(component_key)
                if not state or not state["rids"]:
                    log(f"COMPONENT {component_key}: skip (no matching files)")
                    continue

                package_id = f"{PACKAGE_PREFIX}.{component_key}".lower()
                readme_path = readme_map.get(component_key)
                readme_pack_path = None
                if readme_path:
                    readme_pack_path = work_dir / "README.md"
                    shutil.copy2(readme_path, readme_pack_path)
                    log(
                        f"COMPONENT {component_key}: readme {readme_path.name} -> README.md"
                    )

                log(f"COMPONENT {component_key}: pack (merge)")
                pack_command = [
                    "dotnet",
                    "pack",
                    str(project_path),
                    "--nologo",
                    "--no-build",
                    "--no-restore",
                    f"-p:PackageId={package_id}",
                    f"-p:Version={version}",
                    "-p:PackageRid=",
                    f"-p:PackageComponent={component_key}",
                    f"-p:PackageContentDir={state['content_root']}",
                    "-p:PackageContentPath=",
                    f"-p:PackageOutputPath={artifacts_dir}",
                    "-p:NoWarn=NU5128",
                    "-p:IncludeSymbols=false",
                    "-p:IncludeSource=false",
                ]
                if readme_pack_path:
                    pack_command.extend(
                        [
                            "-p:PackageReadmeFile=README.md",
                            f"-p:PackageReadmePath={readme_pack_path}",
                        ]
                    )
                run_command(pack_command)

                nupkg_path = artifacts_dir / f"{package_id}.{version}.nupkg"
                if not nupkg_path.exists():
                    fail(f"Expected package not found: {nupkg_path}")
                log(f"COMPONENT {component_key}: output {nupkg_path}")
                log_nupkg_tree(nupkg_path)

                built_package_versions.append(f"{package_id}@{version}")
                shutil.rmtree(state["content_root"], ignore_errors=True)

        if not processed_any and not built_package_versions and not skipped_existing:
            fail("No suitable archives found to process")

    if rid_results:
        log("RID packages:")
        for rid, components in sorted(rid_results.items()):
            log(f"RID {rid}:")
            for component, files in sorted(components.items()):
                log(f"  {component}:")
                for file_path in sorted(files):
                    log(f"    {file_path}")

    if built_package_versions:
        log("Built packages:")
        for pkg in built_package_versions:
            log(f"  {pkg}")


if __name__ == "__main__":
    main()
