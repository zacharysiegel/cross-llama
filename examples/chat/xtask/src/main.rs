use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, ExitStatus};

const CARGO_LEPTOS_VERSION: &str = "0.3.10";
const WASM_BINDGEN_PACKAGE_NAME: &str = "wasm-bindgen";
const LEPTOS_TASK: &str = "leptos";
const USAGE: &str = "usage: cargo xtask leptos <cargo-leptos arguments>";
// The conventional status for a misuse of a command.
const USAGE_EXIT_STATUS: u8 = 64;

minimer::define_app_error!(AppError);

/// A package that `cargo install` installs at an exact version.
struct PinnedTool {
    package_name: &'static str,
    version: String,
    executable_name: &'static str,
}

fn main() -> ExitCode {
    let outcome: Result<ExitCode, AppError> = run();

    match outcome {
        Ok(exit_code) => exit_code,
        Err(error) => {
            eprintln!("{}", error.message);
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<ExitCode, AppError> {
    let mut arguments: std::iter::Skip<std::env::ArgsOs> = std::env::args_os().skip(1);
    let task: Option<OsString> = arguments.next();

    if task.as_deref() != Some(OsStr::new(LEPTOS_TASK)) {
        eprintln!("{USAGE}");
        return Ok(ExitCode::from(USAGE_EXIT_STATUS));
    }

    let cargo_leptos_tool: PinnedTool = PinnedTool {
        package_name: "cargo-leptos",
        version: CARGO_LEPTOS_VERSION.to_string(),
        executable_name: "cargo-leptos",
    };
    let wasm_bindgen_tool: PinnedTool = PinnedTool {
        package_name: "wasm-bindgen-cli",
        version: read_locked_wasm_bindgen_version()?,
        executable_name: "wasm-bindgen",
    };

    let cargo_leptos_directory: PathBuf = install_unless_installed(&cargo_leptos_tool)?;
    let wasm_bindgen_directory: PathBuf = install_unless_installed(&wasm_bindgen_tool)?;
    let cargo_leptos_path: PathBuf = cargo_leptos_directory.join(cargo_leptos_tool.executable_name);
    let cargo_leptos_arguments: Vec<OsString> = arguments.collect();

    run_cargo_leptos(&cargo_leptos_path, &wasm_bindgen_directory, &cargo_leptos_arguments)
}

fn get_demo_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The wasm-bindgen command line must be the version of the wasm-bindgen library the web client links.
fn read_locked_wasm_bindgen_version() -> Result<String, AppError> {
    let lock_path: PathBuf = get_demo_directory().join("Cargo.lock");
    let lock_text: String = std::fs::read_to_string(&lock_path)?;
    let locked_version: Option<String> = find_locked_version(&lock_text, WASM_BINDGEN_PACKAGE_NAME);

    locked_version.ok_or_else(|| AppError::new(&format!("{} locks no {WASM_BINDGEN_PACKAGE_NAME} version", lock_path.display())))
}

fn find_locked_version(lock_text: &str, package_name: &str) -> Option<String> {
    let name_line: String = format!("name = \"{package_name}\"");
    let mut lines: std::str::Lines = lock_text.lines();

    lines.find(|line: &&str| *line == name_line)?;

    let version_line: &str = lines.next()?;
    let version: &str = version_line.strip_prefix("version = \"")?.strip_suffix('"')?;

    Some(version.to_string())
}

/// Returns the directory that holds the tool's executables, which is named by the version so that another pin
/// installs anew.
fn install_unless_installed(tool: &PinnedTool) -> Result<PathBuf, AppError> {
    let install_root: PathBuf = get_demo_directory().join("build").join("tools").join(format!("{}-{}", tool.package_name, tool.version));
    let executable_directory: PathBuf = install_root.join("bin");

    if executable_directory.join(tool.executable_name).is_file() {
        return Ok(executable_directory);
    }

    eprintln!("Installing {} {} into {}", tool.package_name, tool.version, install_root.display());

    let install_status: ExitStatus = Command::new(get_cargo_path())
        .arg("install")
        .arg(tool.package_name)
        .arg("--version")
        .arg(format!("={}", tool.version))
        .arg("--locked")
        .arg("--root")
        .arg(&install_root)
        .status()?;

    if !install_status.success() {
        return Err(AppError::new(&format!("installing {} {} failed ({install_status})", tool.package_name, tool.version)));
    }

    Ok(executable_directory)
}

// Cargo sets CARGO for the programs `cargo run` starts.
fn get_cargo_path() -> OsString {
    std::env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"))
}

/// cargo-leptos runs the first wasm-bindgen on the `PATH`, so the pinned one goes first.
fn run_cargo_leptos(cargo_leptos_path: &Path, wasm_bindgen_directory: &Path, cargo_leptos_arguments: &[OsString]) -> Result<ExitCode, AppError> {
    let inherited_path: OsString = std::env::var_os("PATH").unwrap_or_default();
    let search_directories: Vec<PathBuf> = std::iter::once(wasm_bindgen_directory.to_path_buf())
        .chain(std::env::split_paths(&inherited_path))
        .collect();
    let search_path: OsString =
        std::env::join_paths(search_directories).map_err(|error| AppError::from_error("cannot build the PATH", Box::new(error)))?;

    let cargo_leptos_status: ExitStatus = Command::new(cargo_leptos_path)
        .args(cargo_leptos_arguments)
        .env("PATH", search_path)
        .current_dir(get_demo_directory())
        .status()?;

    let exit_code: ExitCode = match cargo_leptos_status.code() {
        Some(0) => ExitCode::SUCCESS,
        Some(code) => ExitCode::from(u8::try_from(code).unwrap_or(u8::MAX)),
        None => ExitCode::FAILURE,
    };

    Ok(exit_code)
}

#[cfg(test)]
mod tests {
    #[test]
    fn find_locked_version_reads_the_named_package_only() {
        let lock_text: &str =
            "[[package]]\nname = \"wasm-bindgen-futures\"\nversion = \"0.4.80\"\n\n[[package]]\nname = \"wasm-bindgen\"\nversion = \"0.2.130\"\n";

        assert_eq!(super::find_locked_version(lock_text, "wasm-bindgen").as_deref(), Some("0.2.130"));
    }
}
