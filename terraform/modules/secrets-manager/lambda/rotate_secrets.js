const AWS = require('aws-sdk');
const crypto = require('crypto');

const secretsManager = new AWS.SecretsManager();

exports.handler = async (event) => {
    console.log('Rotation event:', JSON.stringify(event, null, 2));
    
    const { SecretId, Step, Token } = event;
    
    try {
        switch (Step) {
            case 'createSecret':
                await createSecret(SecretId, Token);
                break;
            case 'setSecret':
                await setSecret(SecretId, Token);
                break;
            case 'testSecret':
                await testSecret(SecretId, Token);
                break;
            case 'finishSecret':
                await finishSecret(SecretId, Token);
                break;
            default:
                throw new Error(`Invalid step: ${Step}`);
        }
        
        return { statusCode: 200, body: 'Rotation completed successfully' };
    } catch (error) {
        console.error('Rotation failed:', error);
        throw error;
    }
};

async function createSecret(secretId, token) {
    console.log('Creating new secret version...');
    
    // Get current secret
    const currentSecret = await secretsManager.getSecretValue({
        SecretId: secretId,
        VersionStage: 'AWSCURRENT'
    }).promise();
    
    const currentData = JSON.parse(currentSecret.SecretString);
    
    // Generate new secrets
    const newSecrets = {
        access_token_secret: generateSecureKey(64),
        refresh_token_secret: generateSecureKey(64),
        encryption_key: generateSecureKey(32)
    };
    
    // Preserve any additional fields
    const newData = { ...currentData, ...newSecrets };
    
    try {
        await secretsManager.putSecretValue({
            SecretId: secretId,
            SecretString: JSON.stringify(newData),
            VersionStages: ['AWSPENDING'],
            ClientRequestToken: token
        }).promise();
        
        console.log('New secret version created successfully');
    } catch (error) {
        if (error.code === 'ResourceExistsException') {
            console.log('Secret version already exists');
        } else {
            throw error;
        }
    }
}

async function setSecret(secretId, token) {
    console.log('Setting secret in service...');
    
    // Get the pending secret
    const pendingSecret = await secretsManager.getSecretValue({
        SecretId: secretId,
        VersionId: token,
        VersionStage: 'AWSPENDING'
    }).promise();
    
    // In a real implementation, you would update your service configuration
    // For now, we'll just validate the secret format
    const secretData = JSON.parse(pendingSecret.SecretString);
    
    if (!secretData.access_token_secret || !secretData.refresh_token_secret || !secretData.encryption_key) {
        throw new Error('Invalid secret format');
    }
    
    console.log('Secret validation completed');
}

async function testSecret(secretId, token) {
    console.log('Testing new secret...');
    
    // Get the pending secret
    const pendingSecret = await secretsManager.getSecretValue({
        SecretId: secretId,
        VersionId: token,
        VersionStage: 'AWSPENDING'
    }).promise();
    
    const secretData = JSON.parse(pendingSecret.SecretString);
    
    // Test the new secrets (in a real implementation, you would test JWT signing/verification)
    try {
        // Simulate JWT operations
        const testPayload = { test: true, timestamp: Date.now() };
        const testSignature = crypto
            .createHmac('sha256', secretData.access_token_secret)
            .update(JSON.stringify(testPayload))
            .digest('hex');
        
        if (!testSignature) {
            throw new Error('Failed to generate test signature');
        }
        
        console.log('Secret test completed successfully');
    } catch (error) {
        console.error('Secret test failed:', error);
        throw error;
    }
}

async function finishSecret(secretId, token) {
    console.log('Finishing secret rotation...');
    
    // Move the AWSPENDING version to AWSCURRENT
    await secretsManager.updateSecretVersionStage({
        SecretId: secretId,
        VersionStage: 'AWSCURRENT',
        ClientRequestToken: token,
        RemoveFromVersionId: await getCurrentVersionId(secretId)
    }).promise();
    
    console.log('Secret rotation completed successfully');
}

async function getCurrentVersionId(secretId) {
    const response = await secretsManager.describeSecret({
        SecretId: secretId
    }).promise();
    
    for (const [versionId, stages] of Object.entries(response.VersionIdsToStages)) {
        if (stages.includes('AWSCURRENT')) {
            return versionId;
        }
    }
    
    throw new Error('No current version found');
}

function generateSecureKey(length) {
    return crypto.randomBytes(length).toString('base64');
}