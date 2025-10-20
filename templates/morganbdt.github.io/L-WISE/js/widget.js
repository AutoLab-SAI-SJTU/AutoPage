/**
 * Main widget class that handles user interactions and display
 */

import { WIDGET_CONFIG, CLASS_MAPPINGS, DATASET_CONFIG } from './config.js';
import { DataManager } from './dataManager.js';
import { AnimationManager } from './animations.js';
import { 
  createElement, 
  createLoadingSpinner, 
  createErrorPlaceholder,
  formatClassName,
  updateRangeInput
} from './utils.js';

export class EnhancedImageWidget {
  constructor() {
    this.currentDataset = 'imagenet';
    this.currentIndex = 0;
    this.currentEpsilon = null;
    this.isAutoscrolling = true;
    this.autoScrollInterval = null;
    this.isTransitioning = false;
    this.isAutoSwitchingDataset = false;
    this.rotationCount = 0;
    this.lastIndex = 0;

    this.animations = new AnimationManager();
    this.dataManager = new DataManager(this.handleLoadingProgress.bind(this));

    this.initialize();
  }

  /**
   * Initializes the widget
   */
  async initialize() {
    this.initializeLoadingState();
    try {
      await this.dataManager.loadAllDatasets();
      this.initializeWidget();
    } catch (error) {
      console.error('Failed to initialize widget:', error);
      this.showError('Failed to load image data. Please refresh the page.');
    }
  }

  /**
   * Creates and shows loading overlay
   */
  initializeLoadingState() {
    this.loadingOverlay = createElement('div', {
      className: 'loading-overlay'
    }, [
      createLoadingSpinner(),
      createElement('div', { className: 'loading-text' }, ['Loading images...']),
      createElement('div', { className: 'progress' }, [
        createElement('div', {
          className: 'progress-bar',
          role: 'progressbar',
          'aria-valuenow': '0',
          'aria-valuemin': '0',
          'aria-valuemax': '100'
        })
      ])
    ]);

    document.querySelector('.image-carousel-container').appendChild(this.loadingOverlay);
  }

  /**
   * Updates loading progress
   * @param {number} progress - Progress value between 0 and 1
   */
  handleLoadingProgress(progress) {
    const progressBar = this.loadingOverlay.querySelector('.progress-bar');
    const percentage = Math.round(progress * 100);
    progressBar.style.width = `${percentage}%`;
    progressBar.setAttribute('aria-valuenow', percentage);
  }

  /**
   * Initializes widget after data is loaded
   */
  initializeWidget() {
    this.loadingOverlay.remove();
    this.setupEventListeners();
    this.updateCurrentEpsilon();
    this.updateVisibleImages();
    this.startAutoScroll();
  }

  /**
   * Stops interval timers
   */
  stopIntervals() {
    if (this.autoScrollInterval) {
      clearInterval(this.autoScrollInterval);
      this.autoScrollInterval = null;
    }
  }

  /**
   * Stops autoscroll functionality
   */
  stopAutoScroll() {
    this.isAutoscrolling = false;
    this.stopIntervals();
    this.rotationCount = 0;  // Reset rotation count when stopping autoscroll
    document.querySelector('.resume-button').classList.add('visible');
  }

  /**
   * Sets up event listeners
   */
  setupEventListeners() {
    // Dataset selection
    document.querySelectorAll('input[name="dataset_radio"]').forEach(radio => {
      radio.addEventListener('change', () => {
        if (!this.isAutoSwitchingDataset) {  // Only handle manual changes
          const dataset = radio.id.replace('dataset_', '');
          this.switchDataset(dataset);
        }
      });
    });

    // Epsilon slider
    const slider = document.getElementById('enhancement_epsilon_range');
    slider.addEventListener('input', () => {
      this.handleEpsilonChange(slider.value);
    });

    // Navigation buttons
    document.getElementById('carousel_prev').addEventListener('click', () => {
      this.navigate(-1);
      this.stopAutoScroll();
    });
    
    document.getElementById('carousel_next').addEventListener('click', () => {
      this.navigate(1);
      this.stopAutoScroll();
    });

    // Resume button
    this.setupResumeButton();
  }

  /**
   * Sets up resume autoscroll button
   */
  setupResumeButton() {
    const resumeBtn = createElement('div', {
      className: 'resume-button'
    }, [
      createElement('input', {
        type: 'button',
        className: 'btn-check',
        id: 'resume_autoscroll'
      }),
      createElement('label', {
        className: 'btn btn-primary',
        for: 'resume_autoscroll',
        style: 'white-space: pre-line'
      }, ['Resume\nautoscroll'])
    ]);

    document.querySelector('.button-container').appendChild(resumeBtn);
    
    document.getElementById('resume_autoscroll').addEventListener('click', () => {
      this.startAutoScroll();
      resumeBtn.classList.remove('visible');
    });
  }

  /**
   * Updates epsilon value and slider
   */
  updateCurrentEpsilon() {
    const config = this.dataManager.getDatasetConfig(this.currentDataset);
    this.currentEpsilon = config.defaultEpsilon;
    
    const slider = document.getElementById('enhancement_epsilon_range');
    const epsilons = config.epsilons;
    const currentIndex = epsilons.indexOf(this.currentEpsilon);
    updateRangeInput(slider, (currentIndex / (epsilons.length - 1)) * 100);
    
    document.getElementById('current_epsilon').textContent = this.currentEpsilon;
  }

  /**
   * Handles epsilon slider change
   * @param {number} value - Slider value
   */
  async handleEpsilonChange(value) {
    const config = this.dataManager.getDatasetConfig(this.currentDataset);
    const epsilons = config.epsilons;
    const index = Math.floor(value * (epsilons.length - 1) / 100);
    const newEpsilon = epsilons[index];

    this.stopAutoScroll();
  
    if (newEpsilon !== this.currentEpsilon) {
      // Start transition before changing epsilon
      const enhancedContainers = document.querySelectorAll('#enhanced_images .image-container');
      
      // Create and position new images on top of existing ones
      const promises = Array.from(enhancedContainers).map(async (container, idx) => {
        const imageData = this.dataManager.getVisibleImages(
          this.currentDataset,
          this.currentIndex,
          WIDGET_CONFIG.VISIBLE_IMAGES
        )[idx];

        if (!imageData) return;

        try {
          const imagePair = await this.dataManager.imageLoader.loadImagePair(
            imageData,
            imageData.index,
            this.currentDataset,
            newEpsilon,
            config.getBucketName
          );

          // Create new image element
          const newImg = imagePair.enhanced.cloneNode(true);
          newImg.style.position = 'absolute';
          newImg.style.top = '0';
          newImg.style.left = '0';
          newImg.style.opacity = '0';
          newImg.style.transition = 'opacity 0.3s ease';
          container.style.position = 'relative';
          container.appendChild(newImg);

          // Trigger fade transition
          await new Promise(resolve => {
            requestAnimationFrame(() => {
              newImg.style.opacity = '1';
              setTimeout(() => {
                // Remove old image after fade
                if (container.children.length > 1) {
                  container.removeChild(container.children[0]);
                }
                resolve();
              }, 300);
            });
          });
        } catch (error) {
          console.error('Error during transition:', error);
        }
      });

      // Wait for all transitions to complete
      await Promise.all(promises);
      
      this.currentEpsilon = newEpsilon;
      document.getElementById('current_epsilon').textContent = this.formatEpsilon(this.currentEpsilon);
    }
  }

  // Helper method to format epsilon values
  formatEpsilon(epsilon) {
    return Number.isInteger(epsilon) ? epsilon : epsilon.toFixed(2);
  }

  /**
   * Updates visible images
   * @param {boolean} skipAnimation - Whether to skip animation
   */
  async updateVisibleImages(skipAnimation = false) {
    if (this.isTransitioning) return;
    this.isTransitioning = true;

    // Store these values to ensure consistency throughout the update
    const currentDatasetSnapshot = this.currentDataset;
    const currentIndexSnapshot = this.currentIndex;
    const currentEpsilonSnapshot = this.currentEpsilon;

    const rowsContainer = document.querySelector('.rows-container');
    const imagesGrid = document.querySelector('.images-grid');

    const originalRow = document.getElementById('original_images');
    const enhancedRow = document.getElementById('enhanced_images');
    
    try {
        const visibleImages = this.dataManager.getVisibleImages(
            currentDatasetSnapshot,
            currentIndexSnapshot,
            WIDGET_CONFIG.VISIBLE_IMAGES
        );

        // Add a check to prevent updates if dataset or index changed during async operations
        if (currentDatasetSnapshot !== this.currentDataset || 
            currentIndexSnapshot !== this.currentIndex) {
            console.log('Dataset or index changed during update, aborting.');
            return;
        }

        // Create new content
        const [newOriginalContent, newEnhancedContent] = await Promise.all([
            this.createImageRow(visibleImages, 'original'),
            this.createImageRow(visibleImages, 'enhanced')
        ]);

        // Check again after async operations
        if (currentDatasetSnapshot !== this.currentDataset || 
            currentIndexSnapshot !== this.currentIndex || 
            currentEpsilonSnapshot !== this.currentEpsilon) {
            console.log('State changed during image creation, aborting update.');
            return;
        }

        if (skipAnimation) {
            // Update content immediately...
            const captionsRow = rowsContainer.querySelector('.captions-row') || 
                              createElement('div', { className: 'captions-row' });
            captionsRow.innerHTML = '';
            this.createCaptions(visibleImages).forEach(caption => 
                captionsRow.appendChild(caption));
            
            originalRow.innerHTML = '';
            enhancedRow.innerHTML = '';
            originalRow.appendChild(newOriginalContent);
            enhancedRow.appendChild(newEnhancedContent);
            
            if (!rowsContainer.querySelector('.captions-row')) {
                rowsContainer.insertBefore(captionsRow, rowsContainer.firstChild);
            }
        } else {
            // Animated update with the same consistency checks...
            await Promise.all([
                this.animations.fade(originalRow, 1, 0, 'fadeOutOriginal'),
                this.animations.fade(enhancedRow, 1, 0, 'fadeOutEnhanced')
            ]);

            // Final check before committing changes
            if (currentDatasetSnapshot !== this.currentDataset || 
                currentIndexSnapshot !== this.currentIndex || 
                currentEpsilonSnapshot !== this.currentEpsilon) {
                console.log('State changed during animation, aborting update.');
                return;
            }

            const captionsRow = rowsContainer.querySelector('.captions-row') || 
                              createElement('div', { className: 'captions-row' });
            captionsRow.innerHTML = '';
            this.createCaptions(visibleImages).forEach(caption => 
                captionsRow.appendChild(caption));
            
            originalRow.innerHTML = '';
            enhancedRow.innerHTML = '';
            originalRow.appendChild(newOriginalContent);
            enhancedRow.appendChild(newEnhancedContent);

            if (!rowsContainer.querySelector('.captions-row')) {
                rowsContainer.insertBefore(captionsRow, rowsContainer.firstChild);
            }

            await Promise.all([
                this.animations.fade(originalRow, 0, 1, 'fadeInOriginal'),
                this.animations.fade(enhancedRow, 0, 1, 'fadeInEnhanced')
            ]);
        }

        // Preload next batch of images
        this.dataManager.preloadUpcoming(
            currentDatasetSnapshot,
            currentIndexSnapshot + WIDGET_CONFIG.VISIBLE_IMAGES,
            WIDGET_CONFIG.VISIBLE_IMAGES
        );
    } catch (error) {
        console.error('Error updating images:', error);
        this.showError('Failed to update images');
    } finally {
        this.isTransitioning = false;
    }
  }

  // New helper method to create captions
  createCaptions(images) {
    return images.map(imageData => {
        const caption = createElement('div', { className: 'image-caption' }, [
            `${formatClassName(imageData.class, this.currentDataset, CLASS_MAPPINGS)}\ndifficulty: ${parseFloat(imageData.difficulty).toFixed(2)}`
        ]);
        return caption;
    });
  }

  /**
   * Creates an image row element
   * @param {Array} images - Array of image data
   * @param {string} type - Row type ('original' or 'enhanced')
   * @returns {Promise<DocumentFragment>}
   */
  async createImageRow(images, type) {
    const fragment = document.createDocumentFragment();
    const config = this.dataManager.getDatasetConfig(this.currentDataset);
    
    for (const imageData of images) {
        const column = createElement('div', { className: 'image-column' });
        const container = createElement('div', { className: 'image-container' });
        
        try {
            // Load both original and enhanced images together
            const imagePair = await this.dataManager.imageLoader.loadImagePair(
                imageData,
                imageData.index,
                this.currentDataset,
                this.currentEpsilon,
                config.getBucketName
            );

            // Use the appropriate image from the pair
            const img = type === 'original' ? imagePair.original : imagePair.enhanced;
            container.appendChild(img.cloneNode(true));
        } catch (error) {
            console.error('Error loading image:', error);
            container.appendChild(createErrorPlaceholder());
        }

        column.appendChild(container);
        fragment.appendChild(column);
    }

    return fragment;
  }

  /**
   * Navigates to next/previous images
   * @param {number} direction - Navigation direction (1 or -1)
   */
  async navigate(direction) {
    if (this.isTransitioning) return;
    
    const dataset = this.currentDataset;
    const datasetLength = this.dataManager.getDataset(dataset).length;
    const newIndex = (this.currentIndex + direction + datasetLength) % datasetLength;
  
    // Only update if we're still on the same dataset
    if (dataset === this.currentDataset) {
      // Detect if we've completed a rotation (gone back to the start)
      if (this.isAutoscrolling && direction > 0) {
        this.rotationCount++;
        
        // Switch dataset after 3 rotations
        if (this.rotationCount >= 3) {
          this.rotationCount = 0; // Reset rotation count
          
          // Find next dataset
          const datasets = Object.keys(DATASET_CONFIG);
          const currentDatasetIndex = datasets.indexOf(this.currentDataset);
          const nextDataset = datasets[(currentDatasetIndex + 1) % datasets.length];
          
          // Trigger dataset switch
          await this.switchDataset(nextDataset, true);
          return;
        }
      }
  
      this.currentIndex = newIndex;
      await this.updateVisibleImages();
    }
  }

  /**
   * Updates the selected radio button for dataset
   * @param {string} dataset - Dataset name
   */
  updateDatasetRadio(dataset) {
    const radio = document.getElementById(`dataset_${dataset}`);
    if (radio) {
      radio.checked = true;
    }
  }

  /**
   * Switches to a different dataset
   * @param {string} dataset - Dataset name
   * @param {boolean} isAutoSwitch - Whether this is an automatic switch
   */
  async switchDataset(dataset, isAutoSwitch = false) {
    if (this.currentDataset === dataset) return;
    
    // Only stop autoscroll if it's a manual switch
    if (!isAutoSwitch) {
      this.stopAutoScroll();
    }
    
    this.isAutoSwitchingDataset = isAutoSwitch;
    this.rotationCount = 0;
    this.lastIndex = 0;
    
    // Clear existing image cache before switching datasets
    // this.dataManager.imageLoader.clearCache();
    
    this.currentDataset = dataset;
    this.currentIndex = 0;
    this.updateCurrentEpsilon();
    
    // Update the radio button selection
    this.updateDatasetRadio(dataset);
    
    await this.updateVisibleImages(true);
    
    this.isAutoSwitchingDataset = false;

    // If this was an auto-switch and autoscroll is enabled, ensure it continues
    if (isAutoSwitch && this.isAutoscrolling) {
      this.startAutoScroll();
    }
  }

  initializeWidget() {
    this.loadingOverlay.remove();
    this.setupEventListeners();
    this.updateCurrentEpsilon();
    this.updateVisibleImages();
    this.startAutoScroll();

    // Update epsilon display format
    this.updateEpsilonDisplay();
  }

  updateEpsilonDisplay() {
    const epsilon = this.currentEpsilon;
    const formattedEpsilon = Number.isInteger(epsilon) ? epsilon : epsilon.toFixed(2);
    document.getElementById('current_epsilon').textContent = formattedEpsilon;
  }

  /**
   * Starts autoscroll functionality
   */
  startAutoScroll() {
    this.isAutoscrolling = true;
    this.rotationCount = 0;
    
    if (this.autoScrollInterval) clearInterval(this.autoScrollInterval);
  
    this.autoScrollInterval = setInterval(() => {
      if (this.isAutoscrolling && !this.isTransitioning) {
        this.navigate(1);
      }
    }, WIDGET_CONFIG.SCROLL_INTERVAL);
  
    document.querySelector('.resume-button').classList.remove('visible');
  }

  /**
   * Stops autoscroll functionality
   */
  stopAutoScroll() {
    this.isAutoscrolling = false;
    if (this.autoScrollInterval) {
      clearInterval(this.autoScrollInterval);
      this.autoScrollInterval = null;
    }
    this.rotationCount = 0;
    document.querySelector('.resume-button').classList.add('visible');
  }

  /**
   * Shows error message
   * @param {string} message - Error message
   */
  showError(message) {
    const container = document.querySelector('.image-carousel-container');
    const errorElement = createElement('div', {
      className: 'error-message'
    }, [message]);
    
    container.innerHTML = '';
    container.appendChild(errorElement);
  }

  /**
   * Cleans up resources
   */
  cleanup() {
    this.stopAutoScroll();
    this.dataManager.cleanup();
    this.animations = null;
  }
}

// Initialize widget when document is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.enhancedImageWidget = new EnhancedImageWidget();
});