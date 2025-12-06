# Authentication Setup

This project implements a minimal token-based authentication system for the Antibody Generation application.

## Server-Side Configuration

### 1. Environment Variable

Set the `ANTIBODY_GEN_ACCESS_TOKEN` environment variable with your desired access token:

```bash
# Linux/macOS
export ANTIBODY_GEN_ACCESS_TOKEN="your-secret-token-here"

# Windows (PowerShell)
$env:ANTIBODY_GEN_ACCESS_TOKEN="your-secret-token-here"

# Windows (Command Prompt)
set ANTIBODY_GEN_ACCESS_TOKEN=your-secret-token-here
```

Alternatively, create a `.env` file in the `agent` directory:

```env
ANTIBODY_GEN_ACCESS_TOKEN=your-secret-token-here
```

### 2. Server Features

- **Authentication Middleware**: All API requests (except `/health` and OPTIONS) require a valid Bearer token
- **Token Verification**: The server validates tokens against the environment variable
- **Error Handling**: Returns proper HTTP 401 responses with WWW-Authenticate headers
- **Logging**: Logs invalid token attempts with client IP addresses

## Client-Side Features

### 1. Authentication Flow

1. **Initial Access**: Users are redirected to `/auth` if no valid token is found
2. **Token Storage**: Tokens are stored securely in browser localStorage
3. **Automatic Redirect**: Users are redirected to `/agents` after successful authentication
4. **Session Persistence**: Tokens persist across browser sessions

### 2. UI Components

- **Auth Page** (`/auth`): Beautiful token input form with validation
- **Protected Routes**: All protected pages require authentication
- **Logout Functionality**: Users can logout via the user dropdown menu
- **Error Handling**: Automatic redirect to auth page on 401 errors

### 3. Security Features

- **Token Validation**: Client-side validation before submission
- **Secure Storage**: Tokens stored in localStorage (consider using httpOnly cookies for production)
- **Automatic Cleanup**: Tokens are removed on logout or 401 errors

## Usage

### 1. Start the Server

```bash
cd agent
uv run main.py
```

### 2. Access the Application

1. Open your browser and navigate to the application
2. You'll be redirected to the authentication page
3. Enter the token that matches your `ANTIBODY_GEN_ACCESS_TOKEN`
4. Click "Authenticate" to proceed to the chat interface

### 3. Making API Requests

The API client automatically includes the Bearer token in all requests:

```typescript
// The token is automatically added to requests
const response = await apiClient.get('/sessions')
```

### 4. Logout

Click on the user avatar in the top-right corner and select "Logout" to clear your token and return to the auth page.

## Security Considerations

### Production Recommendations

1. **HTTPS**: Always use HTTPS in production
2. **Secure Cookies**: Consider using httpOnly cookies instead of localStorage
3. **Token Rotation**: Implement token rotation for enhanced security
4. **Rate Limiting**: Add rate limiting to prevent brute force attacks
5. **Audit Logging**: Enhance logging for security monitoring

### Environment Variables

- Keep your access token secure and never commit it to version control
- Use different tokens for development, staging, and production environments
- Consider using a secrets management service for production deployments

## Troubleshooting

### Common Issues

1. **401 Unauthorized**: Check that your environment variable is set correctly
2. **Token Not Saved**: Ensure localStorage is enabled in your browser
3. **Redirect Loop**: Clear localStorage and restart the authentication flow

### Debug Mode

To disable authentication for development, simply don't set the `ANTIBODY_GEN_ACCESS_TOKEN` environment variable. The server will allow all requests without authentication.
