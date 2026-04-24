# Frontend Dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY frontend/package.json frontend/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy application code
COPY frontend/ .

# Build the app
RUN npm run build

# Expose port
EXPOSE 5173

# Run the development server (or use `npm run preview` for production)
CMD ["npm", "run", "dev", "--", "--host"]
