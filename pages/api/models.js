import fs from 'fs';
import path from 'path';
import { getServerSession } from 'next-auth';
import { authOptions } from '@/pages/auth/[...nextauth]';
import dbConnect from '@/lib/dbConnect';
import { generateApiCode } from '@/templates/ApiCompose';
import { generateePageCode } from '@/templates/pageCompose';
import { generateModelCode } from '@/templates/ModelCompose';
import { generateComponentCode } from '@/templates/componentCompose';
import { generateCreateFormCode } from '@/templates/CreatePage';
import { generateEditFormCode } from '@/templates/EditPage';
import { generateApiWithTokenSupport } from '@/templates/extenalApiCompose';

// Simple file-based store to avoid mongoose dependency
const STORE_PATH = path.join(process.cwd(), 'config', 'models-store.json');

function readStore() {
  try {
    if (!fs.existsSync(STORE_PATH)) {
      return [];
    }
    const raw = fs.readFileSync(STORE_PATH, 'utf-8');
    return JSON.parse(raw || '[]');
  } catch (e) {
    console.warn('Failed to read models store:', e);
    return [];
  }
}

function writeStore(models) {
  try {
    const dir = path.dirname(STORE_PATH);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(STORE_PATH, JSON.stringify(models, null, 2), 'utf-8');
  } catch (e) {
    console.error('Failed to write models store:', e);
  }
}

try {
  const needsRestartPath = path.join(process.cwd(), '.needs-restart');
  if (fs.existsSync(needsRestartPath)) {
    fs.unlinkSync(needsRestartPath);
  }
} catch (error) {
  console.warn('Error ! could not remove .needs-restart file:', error);
}

const checkAuth = async (req, res) => {
  const session = await getServerSession(req, res, authOptions);
  if (!session) {
    res.status(401).json({ message: 'Unauthorized' });
    return false;
  }
  return session;
};

const capitalizeFirstLetter = (str) =>
  typeof str === 'string' && str.length ? str.charAt(0).toUpperCase() + str.slice(1) : str;

export default async function handler(req, res) {
  // Initialize Scylla/Foundation connections (no mongoose)
  await dbConnect();

  const isSignUpUserCreation =
    req.method === 'POST' &&
    req.body.modelType &&
    req.body.modelType.toLowerCase() === 'user' &&
    req.body.data &&
    req.headers.authorization;

  let session = null;
  if (!isSignUpUserCreation) {
    session = await checkAuth(req, res);
    if (!session) return;
  }

  const capitalizeType = (type) => {
    const mapping = { string: 'String', number: 'Number', boolean: 'Boolean', date: 'Date', object: 'Object', array: 'Array' };
    return mapping[type] || 'String';
  };

  const addReverseRelation = (modelName, relatedModelName) => {
    const relatedModelPath = path.join(process.cwd(), 'models', `${capitalizeFirstLetter(relatedModelName)}.js`);
    const reverseField = modelName.toLowerCase() + 's';

    if (!fs.existsSync(relatedModelPath)) return;
    let content = fs.readFileSync(relatedModelPath, 'utf-8');
    if (content.includes(`${reverseField}: [`)) return;

    const insertPattern = /new Schema\s*\(\s*{([\s\S]*?)}\s,\s*{timestamps: true}/;
    const match = insertPattern.exec(content);
    if (!match) return;

    const schemaBody = match[1];
    const lines = schemaBody.trim().split('\n');
    const lastLineIndex = lines.length - 1;
    lines[lastLineIndex] = lines[lastLineIndex].trim().replace(/,?$/, ',');

    const newLine = `${reverseField}: [{ type: Schema.Types.ObjectId, ref: '${capitalizeFirstLetter(modelName)}'  }],`;
    lines.push(newLine);

    const updatedSchemaBody = lines.join('\n');
    const updatedContent = content.replace(schemaBody, updatedSchemaBody);
    fs.writeFileSync(relatedModelPath, updatedContent);
  };

  const formatFields = (fields) => {
    const defaultSeoFields = [
      { name: 'seoTitle', type: 'string', datatype: 'textinput', required: false },
      { name: 'seoDescription', type: 'string', datatype: 'textarea', required: false },
      { name: 'focusKeywords', type: 'array', datatype: 'creatableselectmulti', required: false },
      { name: 'canonicalUrl', type: 'string', datatype: 'stringweblink', required: false },
      { name: 'metaRobots', type: 'string', datatype: 'singleselect', required: false },
      { name: 'openGraphTitle', type: 'string', datatype: 'textinput', required: false },
      { name: 'openGraphDescription', type: 'string', datatype: 'textarea', required: false },
    ];

    const allFields = [...fields];
    defaultSeoFields.forEach((seoField) => {
      if (!fields.some((f) => f.name === seoField.name)) {
        allFields.push(seoField);
      }
    });

    return allFields
      .map((field) => {
        let typeDefinations = `{ type: ${capitalizeType(field.type)} }`;
        const typeProps = [`type: ${capitalizeType(field.type)}`];

        if (field.required) {
          typeProps.push('required: true');
        }

        if (field.datatype) {
          typeProps.push(`datatype: "${field.datatype}"`);
        }

        if (field.enumValues) {
          typeProps.push(`enum:  [${field.enumValues.map((v) => `"${v}"`).join(',')}]`);
        }

        if (field.type === 'array') {
          if (field.refModel) {
            typeDefinations = `[{type: Schema.Types.ObjectId, ref: '${field.refModel}'}]`;
          } else {
            const arrayTypeProps = ['types: String'];
            if (field.datatype) {
              arrayTypeProps.push(`datatype: "${field.datatype}"`);
            }
            if (field.enumValues) {
              arrayTypeProps.push(`enum: [${field.enumValues.map((v) => `${v}`).join(', ')}]`);
            }
            typeDefinations = `[{ ${arrayTypeProps.join(', ')} }]`;
          }
        } else if (field.refModel) {
          typeDefinations = `[{ type: Schema.Types.objectId, ref: '${field.refModel}'}]`;
        } else {
          typeDefinations = `{ ${typeProps.join(', ')} }`;
        }

        return `${field.name}: ${typeDefinations}`;
      })
      .join(',\n');
  };

  // Create model definition and generate files
  if (req.method === 'POST') {
    try {
      let modelName, fields;

      if (req.body.modelType && req.body.data) {
        modelName = req.body.modelType.toLowerCase();
        const userData = req.body.data;

        const userFieldTypes = {
          id: { fieldType: 'string', datatype: 'textinput' },
          firstName: { fieldType: 'string', datatype: 'textinput' },
          lastName: { fieldType: 'string', datatype: 'textinput' },
          email: { fieldType: 'string', datatype: 'textemail' },
          password: { fieldType: 'string', datatype: 'password' },
          userRole: {
            fieldType: 'string',
            datatype: 'singleselect',
            enumValues: ['superadmin', 'contentmanager', 'demo'],
          },
          createdAt: { fieldType: 'date', datatype: 'inputdate' },
          updatedAt: { fieldType: 'date', datatype: 'inputdate' },
        };

        fields = Object.entries(userData).map(([key, value]) => {
          let fieldType, datatype, enumValues;

          if (modelName === 'user' && userFieldTypes[key]) {
            fieldType = userFieldTypes[key].fieldType;
            datatype = userFieldTypes[key].datatype;
            if (userFieldTypes[key].enumValues) {
              enumValues = userFieldTypes[key].enumValues;
            }
          } else {
            const type = Array.isArray(value) ? 'array' : typeof value;

            if (Array.isArray(value)) {
              fieldType = 'array';
              datatype = 'selectmulti';
            } else if (value instanceof Date || (typeof value === 'string' && !isNaN(Date.parse(value)))) {
              fieldType = 'date';
              datatype = 'inputdate';
            } else if (type === 'boolean') {
              fieldType = 'boolean';
              datatype = 'toggleinput';
            } else if (type === 'number') {
              fieldType = 'number';
              datatype = 'number';
            } else {
              fieldType = 'string';
              datatype = 'textinput';
            }
          }

          const requiredUserFields = ['email', 'firstname', 'lastname', 'password', 'userRole'];
          const isRequired =
            modelName === 'user'
              ? requiredUserFields.includes(key)
              : key === 'email' || key === 'name' || key === 'title';

          const field = { name: key, type: fieldType, datatype, required: isRequired };
          if (enumValues) field.enumValues = enumValues;
          return field;
        });

        if (modelName === 'user') {
          const defaultUserFields = [
            { name: 'password', type: 'string', datatype: 'password', required: true },
            {
              name: 'userRole',
              type: 'string',
              datatype: 'singleselect',
              required: true,
              enumValues: ['superadmin', 'contentmanager', 'demo'],
            },
            { name: 'block', type: 'boolean', datatype: 'toggleinput', required: false },
          ];
          defaultUserFields.forEach((defField) => {
            if (!fields.some((field) => field.name === defField.name)) {
              fields.push(defField);
            }
          });
        }
      } else {
        modelName = (req.body.modelName || '').toLowerCase();
        fields = req.body.fields;
      }

      if (!modelName || !fields) {
        return res
          .status(400)
          .json({ message: 'Missing Required Data! modelName & fields are required!' });
      }

      const formattedFields = formatFields(fields);
      const modelContent = generateModelCode(modelName, formattedFields);
      const apiContent = generateApiCode(modelName, fields);
      const externalApicontent = generateApiWithTokenSupport(modelName, fields);
      const pageContent = generateePageCode(modelName, fields);
      const componentContent = generateComponentCode(modelName, fields);
      const createPageContent = generateCreateFormCode(modelName);
      const editPageContent = generateEditFormCode(modelName);

      const ModelName = capitalizeFirstLetter(modelName);
      const modelPath = path.join(process.cwd(), 'models', `${ModelName}.js`);
      const apiPath = path.join(process.cwd(), 'pages', 'api', `${modelName.toLowerCase()}.js`);
      const extApiPath = path.join(process.cwd(), 'pages', 'api', 'public', `${modelName.toLowerCase()}.js`);
      const pageFolderPath = path.join(process.cwd(), 'pages', 'manager', modelName.toLowerCase());
      const editPagePath = path.join(pageFolderPath, 'edit', '[...id].js');
      const componentPath = path.join(process.cwd(), 'components', `${ModelName}.js`);

      // Ensure directories exist
      if (!fs.existsSync(path.dirname(modelPath))) fs.mkdirSync(path.dirname(modelPath), { recursive: true });
      if (!fs.existsSync(path.dirname(apiPath))) fs.mkdirSync(path.dirname(apiPath), { recursive: true });
      if (!fs.existsSync(path.dirname(extApiPath))) fs.mkdirSync(path.dirname(extApiPath), { recursive: true });
      if (!fs.existsSync(pageFolderPath)) fs.mkdirSync(pageFolderPath, { recursive: true });
      if (!fs.existsSync(path.dirname(editPagePath))) fs.mkdirSync(path.dirname(editPagePath), { recursive: true });
      if (!fs.existsSync(path.dirname(componentPath))) fs.mkdirSync(path.dirname(componentPath), { recursive: true });

      // Write files
      fs.writeFileSync(modelPath, modelContent);
      fs.writeFileSync(apiPath, apiContent);
      fs.writeFileSync(extApiPath, externalApicontent);
      fs.writeFileSync(path.join(pageFolderPath, 'index.js'), pageContent);
      fs.writeFileSync(path.join(pageFolderPath, 'create.js'), createPageContent);
      fs.writeFileSync(editPagePath, editPageContent);
      fs.writeFileSync(componentPath, componentContent);

      // Update reverse relation hints in related models if needed
      fields.forEach((field) => {
        if (field.refModel) {
          addReverseRelation(modelName, field.refModel);
        }
      });

      // Persist to file store (no mongoose)
      const store = readStore();
      const id = modelName.toLowerCase();
      const existingIdx = store.findIndex((m) => (m.id || m.name) === id);
      const toSave = { id, name: id, fields };
      if (existingIdx >= 0) {
        store[existingIdx] = toSave;
      } else {
        store.push(toSave);
      }
      writeStore(store);

      return res.status(201).json({ message: 'Model, API & Pages generated successfully!' });
    } catch (err) {
      console.error('Error Generating Models:', err);
      return res.status(500).json({ message: 'Error Generating Models', err: String(err?.message || err) });
    }
  }

  if (req.method === 'PUT') {
    try {
      const { id } = req.query;
      const { modelName, fields } = req.body;
      if (!modelName || !fields) {
        return res.status(400).json({ message: 'modelName and fields are required' });
      }

      const modelNameLower = modelName.toLowerCase();
      const formattedFields = formatFields(fields);

      const store = readStore();
      const idx = store.findIndex((m) => (m.id || m.name) === id);
      if (idx === -1) {
        return res.status(404).json({ message: 'Model Not Found' });
      }

      const existing = store[idx];
      const oldName = (existing.name || existing.id).toLowerCase();
      const OldModelName = capitalizeFirstLetter(oldName);
      const newModelName = capitalizeFirstLetter(modelNameLower);

      const oldmodelPath = path.join(process.cwd(), 'models', `${OldModelName}.js`);
      const oldapiPath = path.join(process.cwd(), 'pages', 'api', `${OldModelName.toLowerCase()}.js`);
      const oldextApiPath = path.join(process.cwd(), 'pages', 'api', 'public', `${OldModelName.toLowerCase()}.js`);
      const oldpageFolderPath = path.join(process.cwd(), 'pages', 'manager', OldModelName.toLowerCase());
      const oldcomponentPath = path.join(process.cwd(), 'components', `${OldModelName}.js`);

      if (oldName !== modelNameLower) {
        if (fs.existsSync(oldmodelPath)) fs.unlinkSync(oldmodelPath);
        if (fs.existsSync(oldapiPath)) fs.unlinkSync(oldapiPath);
        if (fs.existsSync(oldextApiPath)) fs.unlinkSync(oldextApiPath);
        if (fs.existsSync(oldcomponentPath)) fs.unlinkSync(oldcomponentPath);
        if (fs.existsSync(oldpageFolderPath)) fs.rmSync(oldpageFolderPath, { recursive: true, force: true });
      }

      const modelContent = generateModelCode(modelNameLower, formattedFields);
      const apiContent = generateApiCode(modelNameLower, fields);
      const externalApicontent = generateApiWithTokenSupport(modelNameLower, fields);
      const pageContent = generateePageCode(modelNameLower, fields);
      const componentContent = generateComponentCode(modelNameLower, fields);
      const createPageContent = generateCreateFormCode(modelNameLower);
      const editPageContent = generateEditFormCode(modelNameLower);

      const modelPath = path.join(process.cwd(), 'models', `${newModelName}.js`);
      const apiPath = path.join(process.cwd(), 'pages', 'api', `${newModelName}.js`);
      const extApiPath = path.join(process.cwd(), 'pages', 'api', 'public', `${newModelName}.js`);
      const pageFolderPath = path.join(process.cwd(), 'pages', 'manager', newModelName);
      const createPagePath = path.join(pageFolderPath, 'create.js');
      const editFolderPath = path.join(pageFolderPath, 'edit');
      const editPagePath = path.join(editFolderPath, '[...id].js');
      const componentPath = path.join(process.cwd(), 'components', `${newModelName}.js`);

      if (!fs.existsSync(pageFolderPath)) fs.mkdirSync(pageFolderPath, { recursive: true });
      if (!fs.existsSync(editFolderPath)) fs.mkdirSync(editFolderPath, { recursive: true });
      if (!fs.existsSync(path.dirname(modelPath))) fs.mkdirSync(path.dirname(modelPath), { recursive: true });
      if (!fs.existsSync(path.dirname(apiPath))) fs.mkdirSync(path.dirname(apiPath), { recursive: true });
      if (!fs.existsSync(path.dirname(extApiPath))) fs.mkdirSync(path.dirname(extApiPath), { recursive: true });
      if (!fs.existsSync(path.dirname(componentPath))) fs.mkdirSync(path.dirname(componentPath), { recursive: true });

      fs.writeFileSync(modelPath, modelContent);
      fs.writeFileSync(apiPath, apiContent);
      fs.writeFileSync(extApiPath, externalApicontent);
      fs.writeFileSync(path.join(pageFolderPath, 'index.js'), pageContent);
      fs.writeFileSync(createPagePath, createPageContent);
      fs.writeFileSync(editPagePath, editPageContent);
      fs.writeFileSync(componentPath, componentContent);

      fields.forEach((field) => {
        if (field.refModel) {
          addReverseRelation(modelNameLower, field.refModel);
        }
      });

      store[idx] = { id: modelNameLower, name: modelNameLower, fields };
      writeStore(store);

      return res.status(200).json({
        message: 'Model Updated & Files Regenerated Successfully!',
      });
    } catch (err) {
      console.error('Error Updating Model:', err);
      return res.status(500).json({ message: 'Error Updating Models', err: String(err?.message || err) });
    }
  }

  if (req.method === 'DELETE') {
    try {
      const { id } = req.query;

      const store = readStore();
      const idx = store.findIndex((m) => (m.id || m.name) === id);
      if (idx === -1) return res.status(404).json({ message: 'Model Not Found!' });

      const model = store[idx];
      const nameLower = (model.name || model.id).toLowerCase();
      const ModelName = capitalizeFirstLetter(nameLower);

      const modelPath = path.join(process.cwd(), 'models', `${ModelName}.js`);
      const apiPath = path.join(process.cwd(), 'pages', 'api', `${nameLower}.js`);
      const extApiPath = path.join(process.cwd(), 'pages', 'api', 'public', `${nameLower}.js`);
      const pageFolderPath = path.join(process.cwd(), 'pages', 'manager', nameLower);
      const componentPath = path.join(process.cwd(), 'components', `${ModelName}.js`);

      [modelPath, apiPath, extApiPath, componentPath].forEach((file) => {
        if (fs.existsSync(file)) fs.unlinkSync(file);
      });
      if (fs.existsSync(pageFolderPath)) fs.rmSync(pageFolderPath, { recursive: true, force: true });

      store.splice(idx, 1);
      writeStore(store);

      return res.status(200).json({ message: 'Model, API, Pages & component deleted successfully!' });
    } catch (error) {
      console.error('Error Deleting Model:', error);
      return res.status(500).json({ message: 'Error Deleting Model!', error: String(error?.message || error) });
    }
  }

  if (req.method === 'GET') {
    try {
      const { id, model } = req.query;
      const store = readStore();

      if (id) {
        const found = store.find((m) => (m.id || m.name) === id);
        if (!found) return res.status(404).json({ message: 'Model Not Found!' });
        return res.status(200).json(found);
      } else if (model) {
        const found = store.find((m) => (m.name || m.id).toLowerCase() === String(model).toLowerCase());
        if (!found) return res.status(404).json({ message: 'No Model Found With The Given Name!' });
        return res.status(200).json(found);
      } else {
        return res.status(200).json(store);
      }
    } catch (error) {
      console.error('Error Fetching Models:', error);
      return res.status(500).json({ message: 'Error Fetching Models', error: String(error?.message || error) });
    }
  }

  return res.status(405).json({ message: 'Method Not Allowed' });
}

  