# Troubleshooting Guide for Streamlit App Crashes and Disconnections

## Common Issues and Solutions

### 1. **Missing Dependencies (Import Errors)**

**Problem**: Linter shows import errors for packages like `streamlit`, `langchain`, etc.

**Solution**: 
```bash
# Run the installation script
python install_dependencies.py

# Or install manually
pip install -r requirements.txt
```

### 2. **PDF Upload Crashes**

**Problem**: App crashes when uploading large PDF files or during PDF processing.

**Causes**:
- File too large (>50MB)
- Memory issues during vector store creation
- Ollama service not running
- Corrupted PDF files

**Solutions**:
- ✅ **File Size Limit**: PDFs are now limited to 50MB
- ✅ **Memory Management**: Improved batch processing with smaller chunks
- ✅ **Error Handling**: Better cleanup and error recovery
- ✅ **Progress Tracking**: Added delays to prevent frontend disconnection

**Manual Fix**:
```bash
# Ensure Ollama is running
ollama serve

# Install required models
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### 3. **CSV Upload Issues**

**Problem**: CSV processing fails or crashes the app.

**Causes**:
- Large CSV files (>10MB)
- Too many rows (>10,000)
- Encoding issues
- Memory problems

**Solutions**:
- ✅ **File Size Limit**: CSVs limited to 10MB
- ✅ **Row Limit**: Maximum 10,000 rows processed
- ✅ **Batch Processing**: Documents added in smaller batches
- ✅ **Error Recovery**: Skips problematic rows instead of failing

### 4. **Streamlit Disconnections**

**Problem**: Frontend disconnects during file processing.

**Causes**:
- Long-running operations blocking the main thread
- Memory exhaustion
- Network timeouts

**Solutions**:
- ✅ **Progress Indicators**: Added spinners and status updates
- ✅ **Batch Processing**: Smaller chunks with delays
- ✅ **Timeout Handling**: Added timeout context managers
- ✅ **Memory Limits**: File size and row count restrictions

### 5. **Ollama Connection Issues**

**Problem**: "Ollama service not available" errors.

**Solutions**:
```bash
# Start Ollama service
ollama serve

# Check if models are available
ollama list

# Install required models
ollama pull nomic-embed-text
ollama pull llama3:8b
```

### 6. **Memory Issues**

**Problem**: App runs out of memory during processing.

**Solutions**:
- ✅ **File Size Limits**: 50MB for PDFs, 10MB for CSVs
- ✅ **Row Limits**: 10,000 rows max for CSVs
- ✅ **Batch Processing**: Smaller document batches
- ✅ **Cleanup**: Automatic temporary file cleanup

### 7. **Text File Encoding Issues**

**Problem**: Text files fail to load due to encoding.

**Solutions**:
- ✅ **UTF-8 Validation**: Added encoding error handling
- ✅ **Error Messages**: Clear feedback on encoding issues
- ✅ **File Reset**: Proper file pointer management

## Performance Optimizations

### 1. **Vector Store Optimization**
- Unique collection names to prevent conflicts
- Smaller batch sizes (5 for PDFs, 50 for CSVs)
- Temporary directories for isolation

### 2. **Memory Management**
- File size validation before processing
- Automatic cleanup of temporary files
- Row limits for large datasets

### 3. **Error Recovery**
- Graceful handling of batch failures
- Skip problematic rows instead of complete failure
- Comprehensive error logging

## Debugging Steps

### 1. **Check Dependencies**
```bash
python install_dependencies.py
```

### 2. **Verify Ollama Setup**
```bash
ollama serve
ollama list
```

### 3. **Test with Small Files**
- Start with small PDFs (<5MB)
- Test with simple CSV files
- Verify text file encoding

### 4. **Monitor Resources**
- Check memory usage during processing
- Monitor CPU usage
- Watch for disk space issues

### 5. **Check Logs**
- Look for error messages in console
- Check Streamlit logs
- Monitor Ollama service logs

## Prevention Tips

1. **File Preparation**:
   - Compress large PDFs before upload
   - Split large CSVs into smaller files
   - Ensure text files are UTF-8 encoded

2. **System Requirements**:
   - At least 4GB RAM available
   - Sufficient disk space for temporary files
   - Stable internet connection for Ollama models

3. **Best Practices**:
   - Process files one at a time
   - Close unused browser tabs
   - Restart the app if experiencing issues

## Emergency Recovery

If the app becomes unresponsive:

1. **Stop the Streamlit process**:
   ```bash
   # Find and kill the process
   pkill -f streamlit
   ```

2. **Clean up temporary files**:
   ```bash
   # Remove temporary directories
   rm -rf /tmp/chroma_*
   rm -rf ./chroma_db
   ```

3. **Restart Ollama**:
   ```bash
   pkill -f ollama
   ollama serve
   ```

4. **Restart the app**:
   ```bash
   streamlit run app.py
   ```

## Support

If issues persist:
1. Check the console for specific error messages
2. Verify all dependencies are installed correctly
3. Ensure Ollama is running and models are available
4. Try with smaller test files first 