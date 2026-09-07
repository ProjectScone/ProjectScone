// Repository tooling only. Python's installed package remains `scone_memory`.
const fs = require('node:fs');
const path = require('node:path');

function pythonLayout(root) {
  const candidates = [
    {project: 'python/memory', source: 'python/memory/src/scone_memory'},
    {project: 'python/scone-memory', source: 'python/scone-memory/scone_memory'},
  ].map(({project, source}) => ({project: path.resolve(root, project), source: path.resolve(root, source)}));
  const projects = candidates.filter(({project}) => fs.existsSync(path.join(project, 'pyproject.toml')));
  if (projects.length !== 1) {
    throw new Error('Python project must exist at exactly one of python/memory or python/scone-memory; finish the directory migration before running this tool.');
  }
  const selected = projects[0];
  if (!fs.existsSync(path.join(selected.source, 'api/__init__.py'))) {
    throw new Error('Python source package is missing from the selected project: ' + selected.source);
  }
  return selected;
}

module.exports = {pythonLayout};
