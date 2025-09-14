# AWS Credentials Setup for Multi-Agent Swarm System

This guide explains the AWS credentials and permissions required for the Multi-Agent Swarm system to work with AWS Bedrock.

## Required AWS Services

The system uses **AWS Bedrock** to access Claude 3.5 Sonnet for all LLM operations across the three specialized systems (RAG, NL2SQL, and Invoice Processing).

## AWS Credentials Required

### 1. Access Keys
You need the following environment variables:

```env
# Required AWS Credentials
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# Optional: For temporary credentials
AWS_SESSION_TOKEN=your_session_token_here
```

### 2. AWS Bedrock Configuration
```env
# Bedrock Model Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.0
```

## How to Get AWS Credentials

### Option 1: AWS IAM User (Recommended for Development)

1. **Create an IAM User**:
   - Go to AWS Console → IAM → Users → Create User
   - Username: `swarm-bedrock-user` (or any name you prefer)
   - Access type: Programmatic access

2. **Attach Required Policies**:
   - `AmazonBedrockFullAccess` (for full Bedrock access)
   - Or create a custom policy with minimal permissions (see below)

3. **Create Access Keys**:
   - Go to Security credentials tab
   - Create access key → Command Line Interface (CLI)
   - Download the credentials file

### Option 2: AWS CLI Configuration

If you have AWS CLI installed:

```bash
# Configure AWS CLI
aws configure

# Enter your credentials when prompted:
# AWS Access Key ID: [your_access_key]
# AWS Secret Access Key: [your_secret_key]
# Default region name: us-east-1
# Default output format: json
```

### Option 3: Environment Variables

Set the environment variables in your system:

**Windows (PowerShell):**
```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_REGION="us-east-1"
```

**Windows (Command Prompt):**
```cmd
set AWS_ACCESS_KEY_ID=your_access_key_here
set AWS_SECRET_ACCESS_KEY=your_secret_key_here
set AWS_REGION=us-east-1
```

**Linux/Mac:**
```bash
export AWS_ACCESS_KEY_ID="your_access_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_key_here"
export AWS_REGION="us-east-1"
```

### Option 4: .env File (Recommended)

Create a `.env` file in your project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=us-east-1

# Optional: For temporary credentials
AWS_SESSION_TOKEN=your_session_token_here

# Bedrock Configuration
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_TEMPERATURE=0.0

# Database Configuration (Docker)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=swarm_db
DB_USER=swarm_user
DB_PASSWORD=swarm_password

# Redis Configuration (Docker)
REDIS_HOST=localhost
REDIS_PORT=6379
```

## Required IAM Permissions

### Minimal Permissions (Custom Policy)

Create a custom IAM policy with these permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels",
                "bedrock:GetFoundationModel"
            ],
            "Resource": "*"
        }
    ]
}
```

### Full Bedrock Access (Easier)

Attach the managed policy: `AmazonBedrockFullAccess`

## AWS Regions

The system is configured for `us-east-1` by default, but you can use any region where Bedrock is available:

- **us-east-1** (N. Virginia) - Default
- **us-west-2** (Oregon)
- **eu-west-1** (Ireland)
- **ap-southeast-1** (Singapore)

## Testing Your Credentials

### 1. Test AWS CLI Access
```bash
# Test basic AWS access
aws sts get-caller-identity

# Test Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

### 2. Test with Health Check
```bash
# Run the health check to test Bedrock integration
python utils/health_check.py
```

### 3. Test with Examples
```bash
# Run examples to test full system
python utils/run_examples.py
```

## Troubleshooting

### Common Issues

1. **"The security token included in the request is invalid"**
   - Check your AWS credentials are correct
   - Ensure the region is correct
   - Verify the access key is active

2. **"Access Denied"**
   - Check IAM permissions for Bedrock
   - Ensure the user has the required policies attached

3. **"Region not supported"**
   - Verify Bedrock is available in your region
   - Update the region in your configuration

4. **"Model not found"**
   - Check the model ID is correct
   - Ensure the model is available in your region

### Debug Steps

1. **Check Environment Variables**:
   ```bash
   # Windows
   echo %AWS_ACCESS_KEY_ID%
   
   # Linux/Mac
   echo $AWS_ACCESS_KEY_ID
   ```

2. **Test AWS Connection**:
   ```bash
   aws sts get-caller-identity
   ```

3. **Check Bedrock Models**:
   ```bash
   aws bedrock list-foundation-models --region us-east-1
   ```

4. **View Detailed Logs**:
   ```bash
   # Enable debug logging
   export LANGCHAIN_VERBOSE=true
   python utils/health_check.py
   ```

## Security Best Practices

### 1. Use IAM Roles (Production)
- For production, use IAM roles instead of access keys
- Attach roles to EC2 instances or containers

### 2. Rotate Credentials Regularly
- Rotate access keys every 90 days
- Use temporary credentials when possible

### 3. Principle of Least Privilege
- Only grant the minimum required permissions
- Use custom policies instead of managed policies when possible

### 4. Environment Variables
- Never commit credentials to version control
- Use `.env` files and add them to `.gitignore`
- Use AWS Secrets Manager for production

### 5. Monitor Usage
- Enable CloudTrail for API monitoring
- Set up billing alerts for Bedrock usage

## Cost Considerations

### Bedrock Pricing
- **Claude 3.5 Sonnet**: ~$3.00 per 1M input tokens, ~$15.00 per 1M output tokens
- **Free Tier**: 5,000 input tokens and 5,000 output tokens per month

### Cost Optimization
- Use appropriate temperature settings (0.0 for deterministic responses)
- Implement caching for repeated queries
- Monitor usage with AWS Cost Explorer

## Next Steps

After setting up AWS credentials:

1. **Start Docker Services**:
   ```bash
   scripts\docker_start.bat
   ```

2. **Test the System**:
   ```bash
   python utils/health_check.py
   ```

3. **Run Examples**:
   ```bash
   python utils/run_examples.py
   ```

4. **Start the Application**:
   ```bash
   python -m src.app
   ```

## Support

If you encounter issues:

1. Check the logs in `logs/swarm.log`
2. Verify your AWS credentials and permissions
3. Ensure Bedrock is available in your region
4. Test with the health check utility

The system will work without AWS credentials for testing the database and Redis components, but all LLM functionality requires valid AWS Bedrock access.
