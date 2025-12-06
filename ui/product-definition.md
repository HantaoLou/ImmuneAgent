# Product Definition
This is an AI chat application which uses interactive conversations and tool calls to perform complex research & analysis tasks.

## Layout
The main layout of this application is as follows:
- A top navigation bar with a logo, a workspace selector on the left, and a user profile on the right.
- A left sidebar with a sessions list and a new session button.
- A main content area with the chat interface and the results of the tool calls.

## User stories

### Logging in
There is a token based login interface with only one input field for token.

### Starting a session
There is a wizard to help users create a new session. The session requires
- Session name
- Session id (auto generated)
- Set of MCP servers
- Models for reasoning, summarizing, etc.
- Knowledge bases
- Whether online search is enabled.

# For developers
## Development guide
- Use react and typescript to build the app.
- Use design token to manage themes, including
  - Colors
  - Typography
  - Spacing
  - Shadows
  - Rounded corners
- Use ant design component library
- Use llamaindex ui library to build the chat interface.
- Use SSE to stream tokens from backend.

## Backend integration
- This is only a frontend application.
- It is packaged as a dist archive and served by backend server.

## Mocking data
Use `service` files for data retrieving. Before backend is integrated, mock data in service files.

## File name conventions
- For service files, use `*-service.ts` as suffix.
- For component files, use `*-component.tsx` as suffix.
- For page content files, use `*-page.tsx` as suffix.
