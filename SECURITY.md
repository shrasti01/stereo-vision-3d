# Security Guidelines

## Project Security

This document outlines security measures for the Stereo Vision 3D Distance Measurement System.

### Input Validation

- **Path Traversal Protection**: All file paths are validated to prevent `../` attacks
- **ROI Validation**: Region of interest inputs are checked for valid format and range
- **Parameter Bounds**: Calibration parameters are constrained to reasonable values
- **File Size Limits**: Images larger than 100MB are rejected to prevent DoS

### Data Protection

- **No Secrets in Code**: API keys, tokens, and passwords must never be committed
- **Gitignore Rules**: Sensitive files (`.env`, `*.pem`, `credentials.json`) are excluded
- **Output Sanitization**: Generated reports don't contain system information

### Usage Guidelines

1. **Run in Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

2. **Never Run as Root/Administrator**

3. **Validate User Input**:
   - ROI coordinates must be non-negative integers
   - Labels must be alphanumeric only
   - Pattern sizes must be between 3-20

4. **File System Security**:
   - Only process images from trusted sources
   - Keep calibration data in secure locations
   - Don't expose output directories via web servers

### Reporting Vulnerabilities

If you discover a security issue, please report it responsibly:
- Do NOT open a public GitHub issue
- Email the maintainer directly
- Include steps to reproduce

### Dependencies

Keep dependencies updated:
```bash
pip install --upgrade -r requirements.txt
```

Known secure versions:
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- PyYAML >= 6.0

### License

This project is for educational purposes. Use responsibly.