import BaseProcessor from './BaseProcessor.js';
import { createWriteStream, createReadStream, unlinkSync, mkdirSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

/**
 * Document Processor - Handles document processing including PDF thumbnails and text extraction
 */
class DocumentProcessor extends BaseProcessor {
  constructor() {
    super();
    
    // Document processing configurations
    this.supportedFormats = ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt'];
    
    // PDF processing config
    this.pdfConfig = {
      thumbnailDPI: 150,
      thumbnailFormat: 'jpg',
      thumbnailQuality: 80,
      maxPages: 10, // Max pages to generate thumbnails for
      textExtractionLimit: 100000 // Max characters to extract
    };
    
    // Text processing config
    this.textConfig = {
      maxSize: 10 * 1024 * 1024, // 10MB max for text files
      encoding: 'utf8'
    };
    
    // Temporary directory for processing
    this.tempDir = join(tmpdir(), 'document-processing');
    this.ensureTempDir();
  }

  /**
   * Ensure temporary directory exists
   */
  ensureTempDir() {
    try {
      mkdirSync(this.tempDir, { recursive: true });
    } catch (error) {
      console.error('Failed to create temp directory:', error);
    }
  }

  /**
   * Process document file
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Processing result
   */
  async process(job) {
    const startTime = Date.now();
    let tempFiles = [];
    
    try {
      // Validate file
      this.validateFile(job);
      
      // Download original document
      console.log(`Downloading document: ${job.key}`);
      const documentBuffer = await this.downloadFile(job.bucket, job.key);
      
      // Save to temporary file
      const extension = this.getFileExtension(job.key);
      const tempInputPath = join(this.tempDir, `input_${job.jobId}.${extension}`);
      await this.writeBufferToFile(documentBuffer, tempInputPath);
      tempFiles.push(tempInputPath);
      
      // Process document with timeout
      const result = await this.executeWithTimeout(
        this.processDocument(tempInputPath, job, extension)
      );
      
      // Update media record
      await this.updateMediaRecord(job, result);
      
      // Log metrics
      this.logProcessingMetrics(job, startTime, result);
      
      return result;
      
    } catch (error) {
      throw this.handleProcessingError(error, job);
    } finally {
      // Clean up temporary files
      this.cleanupTempFiles(tempFiles);
    }
  }

  /**
   * Process document file
   * @param {string} inputPath - Path to input document file
   * @param {Object} job - Processing job
   * @param {string} extension - File extension
   * @returns {Promise<Object>} Processing result
   */
  async processDocument(inputPath, job, extension) {
    const results = {
      metadata: {},
      thumbnails: [],
      textContent: null,
      outputFiles: []
    };

    // Process based on document type
    switch (extension.toLowerCase()) {
      case 'pdf':
        await this.processPDF(inputPath, job, results);
        break;
      case 'txt':
        await this.processTextFile(inputPath, job, results);
        break;
      case 'doc':
      case 'docx':
        await this.processWordDocument(inputPath, job, results);
        break;
      case 'rtf':
        await this.processRTF(inputPath, job, results);
        break;
      case 'odt':
        await this.processODT(inputPath, job, results);
        break;
      default:
        throw new Error(`Unsupported document format: ${extension}`);
    }

    return results;
  }

  /**
   * Process PDF document
   * @param {string} inputPath - Input PDF path
   * @param {Object} job - Processing job
   * @param {Object} results - Results object to populate
   * @returns {Promise<void>}
   */
  async processPDF(inputPath, job, results) {
    try {
      // Get PDF metadata
      const metadata = await this.getPDFMetadata(inputPath);
      results.metadata = metadata;
      
      console.log(`Processing PDF: ${metadata.pages} pages`);
      
      // Generate thumbnails for first few pages
      const pagesToProcess = Math.min(metadata.pages, this.pdfConfig.maxPages);
      
      for (let page = 1; page <= pagesToProcess; page++) {
        try {
          const thumbnail = await this.generatePDFThumbnail(inputPath, page, job);
          results.thumbnails.push(thumbnail);
          results.outputFiles.push(thumbnail.key);
        } catch (error) {
          console.error(`Failed to generate thumbnail for page ${page}:`, error);
        }
      }
      
      // Extract text content
      try {
        const textContent = await this.extractPDFText(inputPath);
        if (textContent && textContent.length > 0) {
          results.textContent = textContent.substring(0, this.pdfConfig.textExtractionLimit);
          results.metadata.wordCount = this.countWords(textContent);
          results.metadata.charCount = textContent.length;
        }
      } catch (error) {
        console.error('Failed to extract PDF text:', error);
      }
      
    } catch (error) {
      console.error('Error processing PDF:', error);
      throw error;
    }
  }

  /**
   * Process text file
   * @param {string} inputPath - Input text file path
   * @param {Object} job - Processing job
   * @param {Object} results - Results object to populate
   * @returns {Promise<void>}
   */
  async processTextFile(inputPath, job, results) {
    try {
      const textContent = await this.readTextFile(inputPath);
      
      results.textContent = textContent;
      results.metadata = {
        encoding: this.textConfig.encoding,
        lineCount: textContent.split('\n').length,
        wordCount: this.countWords(textContent),
        charCount: textContent.length
      };
      
      console.log(`Processed text file: ${results.metadata.lineCount} lines, ${results.metadata.wordCount} words`);
      
    } catch (error) {
      console.error('Error processing text file:', error);
      throw error;
    }
  }

  /**
   * Process Word document
   * @param {string} inputPath - Input document path
   * @param {Object} job - Processing job
   * @param {Object} results - Results object to populate
   * @returns {Promise<void>}
   */
  async processWordDocument(inputPath, job, results) {
    try {
      // Convert to PDF first for thumbnail generation
      const pdfPath = join(this.tempDir, `converted_${job.jobId}.pdf`);
      
      await this.convertToPDF(inputPath, pdfPath);
      
      // Process as PDF
      await this.processPDF(pdfPath, job, results);
      
      // Extract text using alternative method if available
      try {
        const textContent = await this.extractWordText(inputPath);
        if (textContent) {
          results.textContent = textContent.substring(0, this.pdfConfig.textExtractionLimit);
          results.metadata.wordCount = this.countWords(textContent);
          results.metadata.charCount = textContent.length;
        }
      } catch (error) {
        console.error('Failed to extract Word text:', error);
      }
      
      // Clean up converted PDF
      unlinkSync(pdfPath);
      
    } catch (error) {
      console.error('Error processing Word document:', error);
      throw error;
    }
  }

  /**
   * Process RTF document
   * @param {string} inputPath - Input RTF path
   * @param {Object} job - Processing job
   * @param {Object} results - Results object to populate
   * @returns {Promise<void>}
   */
  async processRTF(inputPath, job, results) {
    try {
      // Convert to PDF for thumbnail generation
      const pdfPath = join(this.tempDir, `converted_${job.jobId}.pdf`);
      
      await this.convertToPDF(inputPath, pdfPath);
      await this.processPDF(pdfPath, job, results);
      
      // Clean up converted PDF
      unlinkSync(pdfPath);
      
    } catch (error) {
      console.error('Error processing RTF document:', error);
      throw error;
    }
  }

  /**
   * Process ODT document
   * @param {string} inputPath - Input ODT path
   * @param {Object} job - Processing job
   * @param {Object} results - Results object to populate
   * @returns {Promise<void>}
   */
  async processODT(inputPath, job, results) {
    try {
      // Convert to PDF for thumbnail generation
      const pdfPath = join(this.tempDir, `converted_${job.jobId}.pdf`);
      
      await this.convertToPDF(inputPath, pdfPath);
      await this.processPDF(pdfPath, job, results);
      
      // Clean up converted PDF
      unlinkSync(pdfPath);
      
    } catch (error) {
      console.error('Error processing ODT document:', error);
      throw error;
    }
  }

  /**
   * Get PDF metadata
   * @param {string} pdfPath - Path to PDF file
   * @returns {Promise<Object>} PDF metadata
   */
  async getPDFMetadata(pdfPath) {
    try {
      // Use pdfinfo command if available
      const { stdout } = await execAsync(`pdfinfo "${pdfPath}"`);
      
      const metadata = {};
      const lines = stdout.split('\n');
      
      for (const line of lines) {
        const [key, ...valueParts] = line.split(':');
        if (key && valueParts.length > 0) {
          const value = valueParts.join(':').trim();
          
          switch (key.trim()) {
            case 'Pages':
              metadata.pages = parseInt(value);
              break;
            case 'Title':
              metadata.title = value;
              break;
            case 'Author':
              metadata.author = value;
              break;
            case 'Creator':
              metadata.creator = value;
              break;
            case 'Producer':
              metadata.producer = value;
              break;
            case 'CreationDate':
              metadata.creationDate = value;
              break;
            case 'ModDate':
              metadata.modificationDate = value;
              break;
            case 'Page size':
              metadata.pageSize = value;
              break;
          }
        }
      }
      
      return metadata;
      
    } catch (error) {
      console.error('Failed to get PDF metadata:', error);
      // Return basic metadata
      return {
        pages: 1,
        title: 'Unknown',
        author: 'Unknown'
      };
    }
  }

  /**
   * Generate PDF thumbnail
   * @param {string} pdfPath - Path to PDF file
   * @param {number} page - Page number
   * @param {Object} job - Processing job
   * @returns {Promise<Object>} Thumbnail info
   */
  async generatePDFThumbnail(pdfPath, page, job) {
    const thumbnailPath = join(this.tempDir, `thumb_${job.jobId}_page_${page}.jpg`);
    
    // Use ImageMagick convert command
    const command = `convert -density ${this.pdfConfig.thumbnailDPI} "${pdfPath}[${page - 1}]" -quality ${this.pdfConfig.thumbnailQuality} -resize 300x400 "${thumbnailPath}"`;
    
    await execAsync(command);
    
    // Upload thumbnail
    const thumbnailBuffer = await this.readFileToBuffer(thumbnailPath);
    const thumbnailKey = this.generateProcessedKey(job.key, `thumb_page_${page}`, 'jpg');
    
    const uploadResult = await this.uploadFile(thumbnailBuffer, thumbnailKey, {
      contentType: 'image/jpeg',
      metadata: {
        'original-key': job.key,
        'page-number': page.toString(),
        'document-type': 'pdf',
        'tenant-id': job.tenantId,
        'processing-job-id': job.jobId
      }
    });
    
    // Clean up temp file
    unlinkSync(thumbnailPath);
    
    return {
      page,
      key: thumbnailKey,
      url: uploadResult.url,
      width: 300,
      height: 400,
      fileSize: thumbnailBuffer.length,
      format: 'jpeg'
    };
  }

  /**
   * Extract text from PDF
   * @param {string} pdfPath - Path to PDF file
   * @returns {Promise<string>} Extracted text
   */
  async extractPDFText(pdfPath) {
    try {
      // Use pdftotext command if available
      const { stdout } = await execAsync(`pdftotext "${pdfPath}" -`);
      return stdout;
    } catch (error) {
      console.error('Failed to extract PDF text:', error);
      return '';
    }
  }

  /**
   * Extract text from Word document
   * @param {string} docPath - Path to Word document
   * @returns {Promise<string>} Extracted text
   */
  async extractWordText(docPath) {
    try {
      // Use antiword for .doc files or docx2txt for .docx files
      const extension = docPath.split('.').pop().toLowerCase();
      
      let command;
      if (extension === 'doc') {
        command = `antiword "${docPath}"`;
      } else if (extension === 'docx') {
        command = `docx2txt "${docPath}" -`;
      } else {
        throw new Error(`Unsupported Word format: ${extension}`);
      }
      
      const { stdout } = await execAsync(command);
      return stdout;
    } catch (error) {
      console.error('Failed to extract Word text:', error);
      return '';
    }
  }

  /**
   * Convert document to PDF
   * @param {string} inputPath - Input document path
   * @param {string} outputPath - Output PDF path
   * @returns {Promise<void>}
   */
  async convertToPDF(inputPath, outputPath) {
    try {
      // Use LibreOffice for conversion
      const command = `libreoffice --headless --convert-to pdf --outdir "${this.tempDir}" "${inputPath}"`;
      await execAsync(command);
      
      // LibreOffice creates PDF with same name as input, rename it
      const inputName = inputPath.split('/').pop().replace(/\.[^/.]+$/, '');
      const generatedPDF = join(this.tempDir, `${inputName}.pdf`);
      
      if (generatedPDF !== outputPath) {
        await execAsync(`mv "${generatedPDF}" "${outputPath}"`);
      }
      
    } catch (error) {
      console.error('Failed to convert to PDF:', error);
      throw error;
    }
  }

  /**
   * Read text file
   * @param {string} filePath - Path to text file
   * @returns {Promise<string>} File content
   */
  async readTextFile(filePath) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      const stream = createReadStream(filePath, { encoding: this.textConfig.encoding });
      
      stream.on('data', chunk => {
        chunks.push(chunk);
        // Prevent reading too much data
        if (chunks.join('').length > this.textConfig.maxSize) {
          stream.destroy();
          resolve(chunks.join('').substring(0, this.textConfig.maxSize));
        }
      });
      
      stream.on('end', () => resolve(chunks.join('')));
      stream.on('error', reject);
    });
  }

  /**
   * Count words in text
   * @param {string} text - Text content
   * @returns {number} Word count
   */
  countWords(text) {
    if (!text) return 0;
    return text.trim().split(/\s+/).filter(word => word.length > 0).length;
  }

  /**
   * Write buffer to file
   * @param {Buffer} buffer - Buffer to write
   * @param {string} filePath - File path
   * @returns {Promise<void>}
   */
  async writeBufferToFile(buffer, filePath) {
    return new Promise((resolve, reject) => {
      const stream = createWriteStream(filePath);
      stream.write(buffer);
      stream.end();
      stream.on('finish', resolve);
      stream.on('error', reject);
    });
  }

  /**
   * Read file to buffer
   * @param {string} filePath - File path
   * @returns {Promise<Buffer>} File buffer
   */
  async readFileToBuffer(filePath) {
    return new Promise((resolve, reject) => {
      const chunks = [];
      const stream = createReadStream(filePath);
      
      stream.on('data', chunk => chunks.push(chunk));
      stream.on('end', () => resolve(Buffer.concat(chunks)));
      stream.on('error', reject);
    });
  }

  /**
   * Get file extension from path
   * @param {string} filePath - File path
   * @returns {string} File extension
   */
  getFileExtension(filePath) {
    return filePath.split('.').pop()?.toLowerCase() || '';
  }

  /**
   * Clean up temporary files
   * @param {Array} files - Array of file paths to clean up
   */
  cleanupTempFiles(files) {
    for (const file of files) {
      try {
        if (file && require('fs').existsSync(file)) {
          unlinkSync(file);
        }
      } catch (error) {
        console.error(`Failed to cleanup temp file ${file}:`, error);
      }
    }
  }

  /**
   * Validate document file
   * @param {Object} job - Processing job
   * @returns {boolean} Validation result
   */
  validateFile(job) {
    super.validateFile(job);
    
    const extension = job.key.split('.').pop()?.toLowerCase();
    
    if (!this.supportedFormats.includes(extension)) {
      throw new Error(`Unsupported document format: ${extension}`);
    }
    
    return true;
  }

  /**
   * Get processing capabilities
   * @returns {Object} Processor capabilities
   */
  getCapabilities() {
    return {
      supportedFormats: this.supportedFormats,
      outputFormats: ['jpg', 'txt'],
      features: [
        'pdf-thumbnail-generation',
        'text-extraction',
        'metadata-extraction',
        'document-conversion',
        'multi-page-processing'
      ],
      maxPages: this.pdfConfig.maxPages,
      textExtractionLimit: this.pdfConfig.textExtractionLimit
    };
  }
}

export default DocumentProcessor;