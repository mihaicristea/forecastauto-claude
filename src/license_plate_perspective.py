# src/license_plate_perspective.py
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlateDetection:
    """Data class for plate detection results"""
    corners: np.ndarray  # 4 corner points
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
class PerspectivePlateProcessor:
    """
    Professional license plate processor with perspective-aware detection and replacement.
    Handles plates at any angle with accurate corner detection.
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"PerspectivePlateProcessor initialized on {self.device}")
        
    def process_image(self, image_path: str, logo_text: str = "Forecast AUTO", 
                     style: str = "eu", output_path: Optional[str] = None) -> Image.Image:
        """
        Main processing function - detect and replace license plate with perspective correction
        
        Args:
            image_path: Path to input image
            logo_text: Text to place on the plate
            style: Plate style ('eu', 'ro', 'us')
            output_path: Optional output path to save result
            
        Returns:
            PIL Image with replaced plate
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect plate with corners
            logger.info("🔍 Detecting license plate corners...")
            plate_detections = self.detect_plate_corners(img)
            
            if not plate_detections:
                logger.warning("⚠️ No license plate detected, trying alternative methods...")
                plate_detections = self._fallback_detection(img)
            
            if plate_detections:
                logger.info(f"✅ Found {len(plate_detections)} plate(s)")
                # Process the best detection
                best_plate = max(plate_detections, key=lambda p: p.confidence)
                result = self._replace_plate_perspective(img, best_plate, logo_text, style)
            else:
                logger.warning("❌ No license plate found")
                result = img_rgb
            
            # Convert to PIL
            result_pil = Image.fromarray(result)
            
            # Save if output path provided
            if output_path:
                result_pil.save(output_path, quality=95)
                logger.info(f"💾 Saved result to: {output_path}")
            
            return result_pil
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def detect_plate_corners(self, img: np.ndarray) -> List[PlateDetection]:
        """
        Detect license plate corners using multiple methods
        
        Returns:
            List of PlateDetection objects
        """
        detections = []
        
        # Method 1: Color-based detection with contour analysis
        color_detections = self._detect_by_color_and_shape(img)
        detections.extend(color_detections)
        
        # Method 2: Edge-based detection
        edge_detections = self._detect_by_edges(img)
        detections.extend(edge_detections)
        
        # Method 3: Text region detection
        text_detections = self._detect_by_text_regions(img)
        detections.extend(text_detections)
        
        # Remove duplicates and low confidence detections
        filtered_detections = self._filter_detections(detections)
        
        return filtered_detections
    
    def _detect_by_color_and_shape(self, img: np.ndarray) -> List[PlateDetection]:
        """Detect plates using color characteristics and shape analysis"""
        detections = []
        height, width = img.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Multiple color masks for different plate types
        masks = []
        
        # White/light plates (most common) - refined ranges
        lower_white = np.array([0, 0, 160])  # Higher value threshold
        upper_white = np.array([180, 25, 255])  # Lower saturation threshold
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Exclude red regions (tail lights, brake lights)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        
        # Remove red regions from white mask
        mask_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(mask_red))
        masks.append(('white', mask_white))
        
        # Yellow plates (some EU countries) - avoid orange/red areas
        lower_yellow = np.array([20, 50, 50])  # More restricted yellow range
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        # Also remove red regions from yellow
        mask_yellow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(mask_red))
        masks.append(('yellow', mask_yellow))
        
        for color_name, mask in masks:
            # Morphological operations to connect plate regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
            morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic filtering
                aspect_ratio = w / h if h > 0 else 0
                area = cv2.contourArea(contour)
                
                # Enhanced license plate criteria
                if (2.5 <= aspect_ratio <= 6.0 and  # Typical plate aspect ratios
                    area > 800 and  # Increased minimum area
                    w > 80 and h > 20 and  # Increased minimum dimensions
                    w < width * 0.35 and h < height * 0.12 and  # Tighter maximum dimensions
                    y > height * 0.5 and  # Must be in bottom half (not top)
                    y < height * 0.95):  # But not at very bottom edge
                    
                    # Additional check: avoid regions that are too high up (likely not plates)
                    center_y = y + h/2
                    if center_y < height * 0.6:  # Skip if center is in upper 60%
                        continue
                    
                    # Check if region contains red pixels (likely tail lights)
                    roi_hsv = hsv[y:y+h, x:x+w]
                    red_pixels = cv2.bitwise_or(
                        cv2.inRange(roi_hsv, lower_red1, upper_red1),
                        cv2.inRange(roi_hsv, lower_red2, upper_red2)
                    )
                    red_ratio = np.sum(red_pixels > 0) / (w * h)
                    
                    if red_ratio > 0.1:  # Skip if more than 10% red pixels
                        continue
                    
                    # Try to find exact corners
                    corners = self._find_plate_corners(img, contour, (x, y, w, h))
                    
                    if corners is not None:
                        # Calculate confidence based on multiple factors
                        confidence = self._calculate_confidence(img, corners, aspect_ratio)
                        
                        # Additional confidence penalty for regions too high up
                        position_penalty = max(0, (height * 0.7 - center_y) / (height * 0.2))
                        confidence = confidence * (1 - position_penalty * 0.5)
                        
                        if confidence > 0.3:  # Only accept high confidence detections
                            detection = PlateDetection(
                                corners=corners,
                                confidence=confidence,
                                bbox=(x, y, w, h)
                            )
                            detections.append(detection)
        
        return detections
    
    def _detect_by_edges(self, img: np.ndarray) -> List[PlateDetection]:
        """Detect plates using edge detection and shape analysis"""
        detections = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 75, 75)
        
        # Edge detection with auto-threshold
        v = np.median(filtered)
        lower = int(max(0, (1.0 - 0.33) * v))
        upper = int(min(255, (1.0 + 0.33) * v))
        edges = cv2.Canny(filtered, lower, upper)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        for contour in contours:
            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Look for quadrilaterals
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / h if h > 0 else 0
                
                if 2.0 <= aspect_ratio <= 7.0 and w > 50 and h > 10:
                    corners = self._order_corners(approx.reshape(-1, 2))
                    confidence = self._calculate_confidence(img, corners, aspect_ratio) * 0.9
                    
                    detection = PlateDetection(
                        corners=corners,
                        confidence=confidence,
                        bbox=(x, y, w, h)
                    )
                    detections.append(detection)
        
        return detections
    
    def _detect_by_text_regions(self, img: np.ndarray) -> List[PlateDetection]:
        """Detect plates by finding text-like regions"""
        detections = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use MSER for text region detection
        mser = cv2.MSER_create(min_area=100, max_area=5000)
        regions, _ = mser.detectRegions(gray)
        
        # Group nearby regions
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        
        # Merge overlapping hulls
        merged_hulls = self._merge_overlapping_regions(hulls)
        
        for hull in merged_hulls:
            x, y, w, h = cv2.boundingRect(hull)
            aspect_ratio = w / h if h > 0 else 0
            
            if 2.5 <= aspect_ratio <= 6.0 and w > 60 and h > 15:
                # Try to find plate corners in this region
                roi = img[y:y+h, x:x+w]
                local_corners = self._find_corners_in_roi(roi)
                
                if local_corners is not None:
                    # Convert to global coordinates
                    corners = local_corners + np.array([x, y])
                    confidence = self._calculate_confidence(img, corners, aspect_ratio) * 0.85
                    
                    detection = PlateDetection(
                        corners=corners,
                        confidence=confidence,
                        bbox=(x, y, w, h)
                    )
                    detections.append(detection)
        
        return detections
    
    def _find_plate_corners(self, img: np.ndarray, contour: np.ndarray, 
                           bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Find exact corners of a license plate given a contour"""
        x, y, w, h = bbox
        
        # Extract ROI with padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        
        roi = img[y1:y2, x1:x2]
        
        # Try to find corners in ROI
        corners = self._find_corners_in_roi(roi)
        
        if corners is not None:
            # Convert back to full image coordinates
            corners = corners + np.array([x1, y1])
            return self._order_corners(corners)
        
        # Fallback: use bounding box corners
        return np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
    
    def _find_corners_in_roi(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """Find corners in a region of interest"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Method 1: Harris corners
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Find corner coordinates
        threshold = 0.01 * corners.max()
        corner_coords = np.argwhere(corners > threshold)
        
        if len(corner_coords) >= 4:
            # Use k-means to find 4 corner clusters
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
            _, _, centers = cv2.kmeans(corner_coords.astype(np.float32), 4, None,
                                      criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert from (y,x) to (x,y)
            centers = centers[:, [1, 0]]
            return self._order_corners(centers)
        
        # Method 2: Find strong edges and their intersections
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 50)
        
        if lines is not None and len(lines) >= 4:
            # Find line intersections
            intersections = self._find_line_intersections(lines, roi.shape)
            if len(intersections) >= 4:
                # Select 4 corners that form the largest quadrilateral
                corners = self._select_best_corners(intersections)
                if corners is not None:
                    return corners
        
        return None
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners: top-left, top-right, bottom-right, bottom-left"""
        # Calculate centroid
        center = np.mean(pts, axis=0)
        
        # Sort by angle from center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = pts[order]
        
        # Find top-left (minimum sum of coordinates)
        tl_idx = np.argmin(ordered[:, 0] + ordered[:, 1])
        
        # Reorder starting from top-left
        ordered = np.roll(ordered, -tl_idx, axis=0)
        
        return ordered.astype(np.float32)
    
    def _order_corners_precise(self, pts: np.ndarray) -> np.ndarray:
        """More precise corner ordering for perspective transform"""
        pts = pts.astype(np.float32)
        
        if len(pts) != 4:
            # If not exactly 4 points, use regular ordering
            return self._order_corners(pts)
        
        # Calculate centroid
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        
        # Separate points into quadrants and find the best candidate for each corner
        quadrants = {
            'top_left': [],
            'top_right': [],
            'bottom_left': [], 
            'bottom_right': []
        }
        
        for point in pts:
            x, y = point
            if x < cx and y < cy:
                quadrants['top_left'].append(point)
            elif x >= cx and y < cy:
                quadrants['top_right'].append(point)
            elif x < cx and y >= cy:
                quadrants['bottom_left'].append(point)
            else:
                quadrants['bottom_right'].append(point)
        
        # Select best point from each quadrant
        corners = []
        for quad_name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            quad_points = quadrants[quad_name]
            if not quad_points:
                # No point in this quadrant, use backup method
                return self._order_corners(pts)
            
            if quad_name == 'top_left':
                # Select point with minimum sum of coordinates
                best_point = min(quad_points, key=lambda p: p[0] + p[1])
            elif quad_name == 'top_right':
                # Select point with maximum x, minimum y
                best_point = max(quad_points, key=lambda p: p[0] - p[1])
            elif quad_name == 'bottom_right':
                # Select point with maximum sum of coordinates
                best_point = max(quad_points, key=lambda p: p[0] + p[1])
            else:  # bottom_left
                # Select point with minimum x, maximum y
                best_point = min(quad_points, key=lambda p: p[0] - p[1])
            
            corners.append(best_point)
        
        return np.array(corners, dtype=np.float32)
    
    def _calculate_confidence(self, img: np.ndarray, corners: np.ndarray, aspect_ratio: float) -> float:
        """
        Calculate confidence score for a detected plate based on various heuristics
        
        Args:
            img: Input image
            corners: Detected corners of the plate
            aspect_ratio: Aspect ratio of the detected plate region
            
        Returns:
            Confidence score (0 to 1)
        """
        height, width = img.shape[:2]
        
        # Safely calculate contour area
        try:
            if corners.shape[0] >= 3:  # Need at least 3 points for area
                # Reshape corners to proper contour format for OpenCV
                contour = corners.reshape(-1, 1, 2).astype(np.int32)
                area = cv2.contourArea(contour) / (width * height)
            else:
                # Fallback: use bounding box area
                bbox = cv2.boundingRect(corners.astype(np.int32))
                area = (bbox[2] * bbox[3]) / (width * height)
        except:
            # Another fallback
            bbox = cv2.boundingRect(corners.astype(np.int32))
            area = (bbox[2] * bbox[3]) / (width * height)
        
        # Aspect ratio score - closer to typical plate ratios is better
        ar_diff = min(abs(aspect_ratio - 4.0), abs(aspect_ratio - 3.0))  # Closer to 4:1 or 3:1 is better
        ar_score = max(0, 1.0 - ar_diff / 2.0)  # Normalize and clamp
        
        # Corner sharpness (variability of corner angles)
        try:
            angles = self._corner_angles(corners)
            sharpness = self._angle_variability(angles)
        except:
            sharpness = 0.5  # Default moderate sharpness
        
        # Color consistency (within the plate region)
        try:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, corners.astype(np.int32), 1)
            mean_val = cv2.mean(img, mask=mask)[:3]
            color_consistency = np.exp(-np.std(mean_val) / 50.0)  # Normalize std
        except:
            color_consistency = 0.5  # Default moderate consistency
        
        # Combine factors into a final score
        confidence = float(area * ar_score * sharpness * color_consistency)
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _corner_angles(self, corners: np.ndarray) -> np.ndarray:
        """Calculate angles at the corners of a quadrilateral"""
        pts = corners.reshape(-1, 2)
        
        if len(pts) < 3:
            return np.array([0, 0, 0, 0])  # Default angles
        
        angles = []
        n_pts = len(pts)
        
        for i in range(n_pts):
            p1 = pts[i]
            p2 = pts[(i + 1) % n_pts]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            angle = np.arctan2(dy, dx)
            angles.append(angle)
        
        # Pad to 4 angles if fewer points
        while len(angles) < 4:
            angles.append(0)
        
        return np.array(angles[:4])  # Return max 4 angles
    
    def _angle_variability(self, angles: np.ndarray) -> float:
        """Measure variability of angles (sharpness)"""
        mean_angle = np.mean(angles)
        variance = np.mean((angles - mean_angle) ** 2)
        
        return 1.0 / (1.0 + variance)  # Inverse variance as sharpness measure
    
    def _merge_overlapping_regions(self, hulls: List[np.ndarray]) -> List[np.ndarray]:
        """Merge overlapping or nearby convex hulls using simple distance checking"""
        if len(hulls) <= 1:
            return hulls
        
        merged = []
        used = set()
        
        for i, hull1 in enumerate(hulls):
            if i in used:
                continue
                
            # Get bounding box of current hull
            x1, y1, w1, h1 = cv2.boundingRect(hull1)
            
            # Check for overlaps with other hulls
            group = [hull1]
            used.add(i)
            
            for j, hull2 in enumerate(hulls):
                if j <= i or j in used:
                    continue
                    
                x2, y2, w2, h2 = cv2.boundingRect(hull2)
                
                # Check if bounding boxes overlap or are close
                margin = 20  # Pixels
                if (x1 - margin < x2 + w2 and x2 - margin < x1 + w1 and
                    y1 - margin < y2 + h2 and y2 - margin < y1 + h1):
                    group.append(hull2)
                    used.add(j)
            
            # Merge the group into a single hull
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Combine all points and create new convex hull
                all_points = np.vstack(group)
                merged_hull = cv2.convexHull(all_points)
                merged.append(merged_hull)
        
        return merged
    
    def _hull_to_polygon(self, hull: np.ndarray) -> np.ndarray:
        """Convert convex hull to simplified polygon representation"""
        # Just return the hull points as a simple polygon
        return hull.reshape(-1, 2)
    
    def _polygon_to_hull(self, poly: np.ndarray) -> np.ndarray:
        """Convert polygon points back to convex hull format"""
        return cv2.convexHull(poly.reshape(-1, 1, 2))
    
    def _find_line_intersections(self, lines: np.ndarray, img_shape: Tuple[int, int]) -> List[np.ndarray]:
        """
        Find intersections of lines in Hough space
        
        Args:
            lines: Detected lines (from HoughLines)
            img_shape: Shape of the image (height, width)
            
        Returns:
            List of intersection points (x, y)
        """
        intersections = []
        
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                rho1, theta1 = lines[i][0]
                rho2, theta2 = lines[j][0]
                
                # Convert polar to cartesian coordinates
                x1 = rho1 * np.cos(theta1)
                y1 = rho1 * np.sin(theta1)
                x2 = rho2 * np.cos(theta2)
                y2 = rho2 * np.sin(theta2)
                
                # Calculate intersection
                det = x1 * y2 - x2 * y1
                if abs(det) < 1e-10:  # Parallel lines
                    continue
                
                x_inter = (y2 - y1) / det
                y_inter = (x1 - x2) / det
                
                # Check if intersection is within image bounds
                if (0 <= x_inter < img_shape[1]) and (0 <= y_inter < img_shape[0]):
                    intersections.append(np.array([x_inter, y_inter]))
        
        return intersections
    
    def _fallback_detection(self, img: np.ndarray) -> List[PlateDetection]:
        """
        Fallback detection method using simplified heuristics
        
        Args:
            img: Input image
            
        Returns:
            List of PlateDetection objects
        """
        detections = []
        height, width = img.shape[:2]
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic filtering
            aspect_ratio = w / h if h > 0 else 0
            area = cv2.contourArea(contour)
            
            if 2.0 <= aspect_ratio <= 7.0 and area > 500:
                # Fallback: use bounding box corners
                corners = np.array([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ], dtype=np.float32)
                
                detection = PlateDetection(
                    corners=corners,
                    confidence=0.5,  # Lower confidence for fallback
                    bbox=(x, y, w, h)
                )
                detections.append(detection)
        
        return detections
    
    def _replace_plate_perspective(self, img: np.ndarray, plate_detection: PlateDetection, 
                                   logo_text: str, style: str) -> np.ndarray:
        """
        Replace the detected license plate with a new plate using perspective transformation
        
        Args:
            img: Input image
            plate_detection: Detected plate corners and confidence
            logo_text: Text to place on the new plate
            style: Plate style ('eu', 'ro', 'us')
            
        Returns:
            Image with replaced license plate
        """
        # Ensure corners are properly ordered for perspective transformation
        corners = self._order_corners_precise(plate_detection.corners.reshape(-1, 2))
        h, w = img.shape[:2]
        
        # Calculate the original plate dimensions to maintain aspect ratio
        bbox = cv2.boundingRect(corners.astype(np.int32))
        original_width = bbox[2]
        original_height = bbox[3]
        aspect_ratio = original_width / original_height if original_height > 0 else 4.0
        
        # Define destination points for the perspective transform
        # Use aspect ratio to maintain plate proportions
        if aspect_ratio > 5:  # Wide plate (US style)
            plate_width = 520
            plate_height = 110
        elif aspect_ratio > 3:  # Standard EU plate
            plate_width = 400
            plate_height = 100
        else:  # Square-ish plate
            plate_width = 300
            plate_height = 150
        
        # Destination corners in top-down view (clockwise from top-left)
        dst_corners = np.array([
            [0, 0],                          # top-left
            [plate_width, 0],                # top-right
            [plate_width, plate_height],     # bottom-right
            [0, plate_height]                # bottom-left
        ], dtype=np.float32)
        
        logger.info(f"🔧 Original corners: {corners}")
        logger.info(f"📐 Plate dimensions: {plate_width}x{plate_height} (AR: {aspect_ratio:.2f})")
        
        try:
            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_corners)
            
            # Warp the plate region to the top-down view
            plate_warped = cv2.warpPerspective(img, M, (plate_width, plate_height))
            
            # Create a new plate design
            plate_image = self._create_plate_design(plate_width, plate_height, logo_text, style)
            
            # Enhanced blending for better integration
            # Use the original plate texture as base and overlay new content
            alpha = 0.2  # Keep more of original texture
            beta = 0.8   # Strong overlay of new design
            combined = cv2.addWeighted(plate_warped, alpha, plate_image, beta, 0)
            
            # Apply feathered edges to the new plate for smoother blending
            mask_feather = self._create_feathered_mask(plate_width, plate_height, feather_size=5)
            combined = cv2.bitwise_and(combined, combined, mask=mask_feather)
            
            # Inverse warp the combined image back to original perspective
            M_inv = cv2.getPerspectiveTransform(dst_corners, corners.astype(np.float32))
            result = cv2.warpPerspective(combined, M_inv, (w, h))
            
            # Create precise mask for plate region with anti-aliasing
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(mask, corners.astype(np.int32), 255)
            
            # Apply Gaussian blur for smoother edges
            mask_blurred = cv2.GaussianBlur(mask, (5, 5), 2)
            mask_norm = mask_blurred.astype(np.float32) / 255.0
            
            # Advanced blending with normalized mask
            result_float = result.astype(np.float32)
            img_float = img.astype(np.float32)
            
            # Apply mask to each channel
            for c in range(3):
                img_float[:, :, c] = (img_float[:, :, c] * (1 - mask_norm) + 
                                     result_float[:, :, c] * mask_norm)
            
            final_result = np.clip(img_float, 0, 255).astype(np.uint8)
            
            logger.info("✅ Perspective transformation completed successfully")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Perspective transformation failed: {e}")
            # Fallback to simple replacement
            return self._simple_plate_replacement(img, plate_detection, logo_text, style)
        
    def _create_plate_design(self, width: int, height: int, text: str, style: str) -> np.ndarray:
        """Create a new license plate design"""
        plate_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Select style and draw the plate
        if style == "eu":
            self._draw_eu_plate(plate_image, text)
        elif style == "ro":
            self._draw_ro_plate(plate_image, text)
        elif style == "us":
            self._draw_us_plate(plate_image, text)
        elif style == "uk":
            self._draw_uk_plate(plate_image, text)
        else:
            # Default to EU style
            self._draw_eu_plate(plate_image, text)
        
        return plate_image
    
    def _create_feathered_mask(self, width: int, height: int, feather_size: int = 5) -> np.ndarray:
        """Create a mask with feathered edges"""
        mask = np.ones((height, width), dtype=np.uint8) * 255
        
        # Create feathered border
        for i in range(feather_size):
            alpha = int(255 * (i + 1) / feather_size)
            cv2.rectangle(mask, (i, i), (width - 1 - i, height - 1 - i), alpha, 1)
        
        return mask
    
    def _simple_plate_replacement(self, img: np.ndarray, plate_detection: PlateDetection, 
                                 logo_text: str, style: str) -> np.ndarray:
        """Fallback simple plate replacement"""
        bbox = plate_detection.bbox
        x, y, w, h = bbox
        
        # Create new plate
        plate_image = self._create_plate_design(w, h, logo_text, style)
        
        # Simple overlay
        result = img.copy()
        result[y:y+h, x:x+w] = plate_image
        
        return result
    
    def _draw_eu_plate(self, plate_image: np.ndarray, text: str):
        """Draw EU-style plate"""
        # Blue background
        cv2.rectangle(plate_image, (0, 0), (plate_image.shape[1], plate_image.shape[0]), (0, 51, 153), -1)
        
        # Yellow border
        cv2.rectangle(plate_image, (5, 5), (plate_image.shape[1] - 5, plate_image.shape[0] - 5), (255, 204, 0), 2)
        
        # Text in the center
        self._put_text_centered(plate_image, text, (255, 255, 255), font_scale=1.5, thickness=2)
    
    def _draw_ro_plate(self, plate_image: np.ndarray, text: str):
        """Draw RO-style plate"""
        # Blue background
        cv2.rectangle(plate_image, (0, 0), (plate_image.shape[1], plate_image.shape[0]), (0, 51, 153), -1)
        
        # Yellow border
        cv2.rectangle(plate_image, (5, 5), (plate_image.shape[1] - 5, plate_image.shape[0] - 5), (255, 204, 0), 2)
        
        # Text in the center
        self._put_text_centered(plate_image, text, (255, 255, 255), font_scale=1.5, thickness=2)
    
    def _draw_us_plate(self, plate_image: np.ndarray, text: str):
        """Draw US-style plate"""
        # White background
        cv2.rectangle(plate_image, (0, 0), (plate_image.shape[1], plate_image.shape[0]), (255, 255, 255), -1)
        
        # Blue border
        cv2.rectangle(plate_image, (5, 5), (plate_image.shape[1] - 5, plate_image.shape[0] - 5), (0, 51, 153), 2)
        
        # Text in the center
        self._put_text_centered(plate_image, text, (0, 0, 0), font_scale=1.5, thickness=2)
    
    def _draw_uk_plate(self, plate_image: np.ndarray, text: str):
        """Draw UK-style plate"""
        # White background
        cv2.rectangle(plate_image, (0, 0), (plate_image.shape[1], plate_image.shape[0]), (255, 255, 255), -1)
        
        # Black border
        cv2.rectangle(plate_image, (2, 2), (plate_image.shape[1] - 3, plate_image.shape[0] - 3), (0, 0, 0), 2)
        
        # Text in the center
        self._put_text_centered(plate_image, text, (0, 0, 0), font_scale=1.5, thickness=2)
    
    def _filter_detections(self, detections: List[PlateDetection]) -> List[PlateDetection]:
        """Filter and deduplicate detected plates"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        # Remove low confidence detections
        filtered = [d for d in detections if d.confidence > 0.3]
        
        # Remove duplicates (overlapping detections)
        final_detections = []
        for detection in filtered:
            is_duplicate = False
            for existing in final_detections:
                # Check overlap
                x1, y1, w1, h1 = detection.bbox
                x2, y2, w2, h2 = existing.bbox
                
                # Calculate intersection over union (IoU)
                intersection_area = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                union_area = w1 * h1 + w2 * h2 - intersection_area
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > 0.5:  # High overlap threshold
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                final_detections.append(detection)
        
        return final_detections[:3]  # Return top 3 detections
    
    def _select_best_corners(self, intersections: List[np.ndarray]) -> Optional[np.ndarray]:
        """Select 4 corners that form the largest quadrilateral"""
        if len(intersections) < 4:
            return None
        
        # Convert to numpy array
        points = np.array(intersections)
        
        # Find the 4 points that form the largest convex hull
        if len(points) == 4:
            return self._order_corners(points)
        
        # For more than 4 points, find the 4 corners of the convex hull
        hull = cv2.convexHull(points.astype(np.float32))
        hull_points = hull.reshape(-1, 2)
        
        if len(hull_points) >= 4:
            # If hull has more than 4 points, select 4 corners
            # Simple approach: select points that are furthest apart
            rect = cv2.minAreaRect(hull_points)
            box = cv2.boxPoints(rect)
            return self._order_corners(box)
        
        return None
    
    def _put_text_centered(self, plate_image: np.ndarray, text: str, color: tuple, font_scale: float = 1.0, thickness: int = 2):
        """Put text centered in the plate image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Calculate center position
        h, w = plate_image.shape[:2]
        text_x = (w - text_width) // 2
        text_y = (h + text_height) // 2
        
        # Put text
        cv2.putText(plate_image, text, (text_x, text_y), font, font_scale, color, thickness)