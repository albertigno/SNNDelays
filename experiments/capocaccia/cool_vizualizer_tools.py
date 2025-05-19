import cv2
import numpy as np
from collections import deque

# Constants
# WIDTH, HEIGHT = 800, 600  # Canvas dimensions

WIDTH, HEIGHT = 300, 300  # Canvas dimensions

# Initialize plot history (stores last N timesteps)
max_history = WIDTH // 2  # Adjust based on desired horizontal resolution
history = deque(maxlen=max_history)

# Create a blank canvas
canvas = np.ones((HEIGHT, WIDTH, 3), dtype=np.uint8) * 255  # White background

def draw_mems(fifo, COLORS):
        for timestep, tensor in enumerate(fifo):

            NEURON_COUNT = len(COLORS)
            # Get membrane potentials (flatten the tensor)
            potentials = tensor.numpy().flatten()
            history.append(potentials)
            
            # Clear canvas
            canvas.fill(255)
            
            # Draw grid lines (optional)
            for y in np.linspace(0, HEIGHT, num=10):
                cv2.line(canvas, (0, int(y)), (WIDTH, int(y)), (200, 200, 200), 1)
            
            # Draw each neuron's line plot
            for neuron_idx in range(NEURON_COUNT):
                # Extract history for this neuron
                neuron_potentials = [h[neuron_idx] for h in history]
                
                # Normalize to canvas height (assuming potentials are in [-1, 1])
                y_coords = []
                for i, val in enumerate(neuron_potentials):
                    x = int(i * (WIDTH / max_history))
                    y = int((1 - (val + 1) / 2) * HEIGHT)  # Map [-1, 1] to [0, HEIGHT]
                    y_coords.append((x, y))
                
                # Draw the line
                for i in range(1, len(y_coords)):
                    cv2.line(canvas, y_coords[i-1], y_coords[i], COLORS[neuron_idx], 2)
            
            # Display the current timestep
            cv2.putText(canvas, f"Timestep: {timestep}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Show the plot
            cv2.imshow("Neuron Membrane Potentials", canvas)
            
            # Exit on 'q' key or after processing all timesteps
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break



BACKGROUND_COLOR = (0, 0, 0)       # Black background
SPIKE_COLOR = (255, 255, 255)      # White spikes (or use colors per neuron)

# Initialize spike history (stores last MAX_HISTORY timesteps)
spike_history = deque(maxlen=max_history*2)

# Create a blank canvas (black)
canvas_spk = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

def draw_spks(spike_deque, NEURON_COUNT):
    # Main loop
    for timestep, spike_tensor in enumerate(spike_deque):
        # Flatten spikes (e.g., tensor([[0, 1, ...]]) -> [0, 1, ...])
        spikes = spike_tensor.numpy().flatten().astype(np.uint8)
        spike_history.append(spikes)
        
        # Clear canvas
        canvas.fill(0)
        
        # Draw spikes for each neuron
        for neuron_idx in range(NEURON_COUNT):
            # Calculate y-position (centered per neuron)
            y = int((neuron_idx + 0.5) * (HEIGHT / NEURON_COUNT))
            
            # Draw spikes for this neuron across history
            for t in range(len(spike_history)):
                if spike_history[t][neuron_idx] == 1:  # Check if spike occurred
                    x = WIDTH - len(spike_history) + t  # Scroll left-to-right
                    cv2.circle(canvas, (x, y), 1, SPIKE_COLOR, -1)  # Draw dot
        
        # Add labels (optional)
        cv2.putText(canvas, f"Timestep: {timestep}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow("Spike Raster Plot", canvas)
        
        # Exit on 'q' key
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break



import cv2
import numpy as np
from collections import deque

# Mock prediction stream (replace with your actual predictions)
# Example: ['A', 'B', 'C', 'A', 'E', ...]
prediction_deque = deque(['A', 'B', 'C', 'A', 'E', 'D', 'B', 'C', ...])

# Constants
WIDTH, HEIGHT = 400, 200  # Small window for minimal latency
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 5
FONT_COLOR = (255, 255, 255)  # White
FONT_THICKNESS = 5

# Create a black canvas
canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

for prediction in prediction_deque:
    # Clear canvas
    canvas.fill(0)
    
    # Calculate text size to center it
    (text_width, text_height), _ = cv2.getTextSize(
        prediction, FONT, FONT_SCALE, FONT_THICKNESS
    )
    x = (WIDTH - text_width) // 2
    y = (HEIGHT + text_height) // 2
    
    # Draw the predicted letter
    cv2.putText(
        canvas, prediction, (x, y), 
        FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS
    )
    
    # Display
    cv2.imshow("Predicted Letter", canvas)
    
    # Exit on 'q' key or 20ms delay (~50 FPS)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()