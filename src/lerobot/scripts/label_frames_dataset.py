#!/usr/bin/env python

"""
Script to add frame-level success/failure labels to an existing teleop dataset for reward classifier training.
This creates labels at the frame level, not episode level.
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.patches as patches


class EpisodeSampler(torch.utils.data.Sampler):
    """Sampler for efficiently loading frames from a specific episode."""
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self):
        return len(self.frame_ids)


class FrameLevelLabeler:
    """Interactive frame-level labeling tool."""
    
    def __init__(self, dataset, episode_idx):
        self.dataset = dataset
        self.episode_idx = episode_idx
        
        # Load episode data efficiently using the sampler
        episode_sampler = EpisodeSampler(dataset, episode_idx)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            sampler=episode_sampler,
            num_workers=0,
        )
        
        # Collect all frames
        self.frames = []
        print(f"Loading episode {episode_idx} frames...")
        for batch in dataloader:
            batch_size = len(batch["index"])
            for i in range(batch_size):
                frame_data = {}
                for key, value in batch.items():
                    frame_data[key] = value[i]
                self.frames.append(frame_data)
        
        self.num_frames = len(self.frames)
        print(f"Loaded {self.num_frames} frames for episode {episode_idx}")
        
        # Get camera keys
        self.camera_keys = []
        for key in dataset.meta.camera_keys:
            if key in self.frames[0]:
                self.camera_keys.append(key)
        
        if not self.camera_keys:
            raise ValueError("No camera images found in dataset")
        
        print(f"Found cameras: {self.camera_keys}")
        
        # Initialize labels: 0=failure, 1=success, -1=ignore
        self.labels = np.zeros(self.num_frames, dtype=int)  # Default to failure
        
        # Current state
        self.current_frame = 0
        self.current_label = 0  # Current brush label
        
        # Setup GUI
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the interactive GUI."""
        # Create subplots: 2 camera images side by side, timeline below
        self.fig = plt.figure(figsize=(20, 12))
        
        # Create grid for layout
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5], width_ratios=[1, 1])
        
        # Camera images (side by side)
        self.ax_handeye = self.fig.add_subplot(gs[0, 0])
        self.ax_global = self.fig.add_subplot(gs[0, 1])
        
        # Timeline (spans both columns)
        self.ax_timeline = self.fig.add_subplot(gs[1, :])
        
        self.fig.suptitle(f'Episode {self.episode_idx} Frame Labeling Tool', fontsize=16)
        
        # Image display
        self.update_image()
        
        # Timeline display
        self.setup_timeline()
        
        # Control buttons
        self.setup_controls()
        
        # Frame slider
        self.setup_slider()
        
        # Connect events
        self.setup_events()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # Make room for controls
    
    def setup_timeline(self):
        """Setup the timeline visualization."""
        self.ax_timeline.set_xlim(0, self.num_frames)
        self.ax_timeline.set_ylim(-0.5, 1.5)
        self.ax_timeline.set_xlabel('Frame Index')
        self.ax_timeline.set_ylabel('Label')
        self.ax_timeline.set_title('Frame Labels (Red=Failure, Green=Success, Gray=Ignore)')
        
        # Draw current labels
        self.update_timeline()
        
        # Current frame indicator
        self.frame_line = self.ax_timeline.axvline(x=self.current_frame, color='blue', linewidth=2, label='Current Frame')
        self.ax_timeline.legend()
    
    def update_timeline(self):
        """Update the timeline visualization."""
        self.ax_timeline.clear()
        
        # Color mapping
        colors = {-1: 'gray', 0: 'red', 1: 'green'}
        
        for i, label in enumerate(self.labels):
            color = colors[label]
            self.ax_timeline.bar(i, 1, color=color, alpha=0.6, width=1.0)
        
        self.ax_timeline.set_xlim(0, self.num_frames)
        self.ax_timeline.set_ylim(-0.5, 1.5)
        self.ax_timeline.set_xlabel('Frame Index')
        self.ax_timeline.set_ylabel('Label')
        self.ax_timeline.set_title('Frame Labels (Red=Failure, Green=Success, Gray=Ignore)')
        
        # Current frame indicator
        self.frame_line = self.ax_timeline.axvline(x=self.current_frame, color='blue', linewidth=3)
        
        # Add statistics
        success_count = np.sum(self.labels == 1)
        failure_count = np.sum(self.labels == 0)
        ignore_count = np.sum(self.labels == -1)
        
        stats_text = f'Success: {success_count}, Failure: {failure_count}, Ignore: {ignore_count}'
        self.ax_timeline.text(0.02, 0.98, stats_text, transform=self.ax_timeline.transAxes, 
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update_image(self):
        """Update the displayed images for both cameras."""
        if 0 <= self.current_frame < self.num_frames:
            frame = self.frames[self.current_frame]
            
            # Show current label
            label_text = {-1: 'IGNORE', 0: 'FAILURE', 1: 'SUCCESS'}[self.labels[self.current_frame]]
            label_color = {'IGNORE': 'gray', 'FAILURE': 'red', 'SUCCESS': 'green'}[label_text]
            
            # Display HandEye camera
            if "observation.images.handeye" in self.camera_keys:
                handeye_tensor = frame["observation.images.handeye"]
                if handeye_tensor.dtype == torch.float32:
                    handeye_numpy = (handeye_tensor * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                else:
                    handeye_numpy = handeye_tensor.permute(1, 2, 0).numpy()
                
                self.ax_handeye.clear()
                self.ax_handeye.imshow(handeye_numpy)
                self.ax_handeye.set_title(f'HandEye Camera - Frame {self.current_frame+1}/{self.num_frames}', 
                                         fontsize=12, fontweight='bold')
                self.ax_handeye.axis('off')
            
            # Display Global camera
            if "observation.images.global" in self.camera_keys:
                global_tensor = frame["observation.images.global"]
                if global_tensor.dtype == torch.float32:
                    global_numpy = (global_tensor * 255).type(torch.uint8).permute(1, 2, 0).numpy()
                else:
                    global_numpy = global_tensor.permute(1, 2, 0).numpy()
                
                self.ax_global.clear()
                self.ax_global.imshow(global_numpy)
                self.ax_global.set_title(f'Global Camera - Frame {self.current_frame+1}/{self.num_frames}', 
                                        fontsize=12, fontweight='bold')
                self.ax_global.axis('off')
            
            # Add label info to the figure title
            self.fig.suptitle(f'Episode {self.episode_idx} Frame Labeling Tool - Label: {label_text}', 
                             fontsize=16, color=label_color, fontweight='bold')
    
    def setup_controls(self):
        """Setup control buttons."""
        # Button dimensions
        button_height = 0.04
        button_width = 0.08
        small_width = 0.06
        
        # Row 1: Navigation buttons (bottom)
        y_row1 = 0.02
        self.btn_prev = Button(plt.axes([0.05, y_row1, small_width, button_height]), '←1')
        self.btn_next = Button(plt.axes([0.12, y_row1, small_width, button_height]), '1→')
        self.btn_prev10 = Button(plt.axes([0.19, y_row1, small_width, button_height]), '←10')
        self.btn_next10 = Button(plt.axes([0.26, y_row1, small_width, button_height]), '10→')
        
        # Row 1: Single frame labeling
        self.btn_failure = Button(plt.axes([0.4, y_row1, button_width, button_height]), 'Failure', color='lightcoral')
        self.btn_success = Button(plt.axes([0.49, y_row1, button_width, button_height]), 'Success', color='lightgreen')
        self.btn_ignore = Button(plt.axes([0.58, y_row1, button_width, button_height]), 'Ignore', color='lightgray')
        
        # Clear button
        self.btn_clear = Button(plt.axes([0.67, y_row1, button_width, button_height]), 'Clear All', color='wheat')
        
        # Save and navigation buttons
        self.btn_save = Button(plt.axes([0.76, y_row1, button_width, button_height]), 'Save', color='lightblue')
        self.btn_next_ep = Button(plt.axes([0.85, y_row1, button_width, button_height]), 'Next Episode', color='orange')
        
        # Row 2: Bulk labeling buttons (top)
        y_row2 = y_row1 + 0.05
        self.btn_start_success = Button(plt.axes([0.05, y_row2, button_width + 0.02, button_height]), 'Mark Success Start', color='lightgreen')
        self.btn_end_success = Button(plt.axes([0.16, y_row2, button_width + 0.02, button_height]), 'Mark Success End', color='green')
        self.btn_add_trailing = Button(plt.axes([0.27, y_row2, button_width + 0.02, button_height]), 'Add Trailing 50', color='darkgreen')
        
        self.btn_start_ignore = Button(plt.axes([0.4, y_row2, button_width + 0.02, button_height]), 'Mark Ignore Start', color='lightgray')
        self.btn_end_ignore = Button(plt.axes([0.51, y_row2, button_width + 0.02, button_height]), 'Mark Ignore End', color='gray')
        
        self.btn_start_failure = Button(plt.axes([0.64, y_row2, button_width + 0.02, button_height]), 'Mark Fail Start', color='lightcoral')
        self.btn_end_failure = Button(plt.axes([0.75, y_row2, button_width + 0.02, button_height]), 'Mark Fail End', color='red')
        
        # Initialize marking state
        self.success_start = None
        self.success_end = None
        self.ignore_start = None
        self.ignore_end = None
        self.failure_start = None
        self.failure_end = None
        
        # Episode management
        self.saved = False
        self.next_episode_requested = False
    
    def setup_slider(self):
        """Setup frame navigation slider."""
        ax_slider = plt.axes([0.1, 0.12, 0.8, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames-1, 
                            valinit=self.current_frame, valfmt='%d')
    
    def setup_events(self):
        """Connect GUI events."""
        # Navigation buttons
        self.btn_prev.on_clicked(lambda x: self.navigate(-1))
        self.btn_next.on_clicked(lambda x: self.navigate(1))
        self.btn_prev10.on_clicked(lambda x: self.navigate(-10))
        self.btn_next10.on_clicked(lambda x: self.navigate(10))
        
        # Single frame labeling
        self.btn_failure.on_clicked(lambda x: self.set_label(0))
        self.btn_success.on_clicked(lambda x: self.set_label(1))
        self.btn_ignore.on_clicked(lambda x: self.set_label(-1))
        
        # Utility buttons
        self.btn_clear.on_clicked(self.clear_all_labels)
        self.btn_save.on_clicked(self.save_labels)
        self.btn_next_ep.on_clicked(self.next_episode)
        
        # Bulk labeling - Success
        self.btn_start_success.on_clicked(self.mark_success_start)
        self.btn_end_success.on_clicked(self.mark_success_end)
        self.btn_add_trailing.on_clicked(self.add_trailing_success)
        
        # Bulk labeling - Ignore
        self.btn_start_ignore.on_clicked(self.mark_ignore_start)
        self.btn_end_ignore.on_clicked(self.mark_ignore_end)
        
        # Bulk labeling - Failure
        self.btn_start_failure.on_clicked(self.mark_failure_start)
        self.btn_end_failure.on_clicked(self.mark_failure_end)
        
        self.slider.on_changed(self.on_slider_change)
        
        # Keyboard shortcuts
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Timeline clicking
        self.ax_timeline.figure.canvas.mpl_connect('button_press_event', self.on_timeline_click)
    
    def navigate(self, delta):
        """Navigate frames."""
        new_frame = np.clip(self.current_frame + delta, 0, self.num_frames - 1)
        self.current_frame = new_frame
        self.slider.set_val(self.current_frame)
        self.update_display()
    
    def set_label(self, label):
        """Set label for current frame."""
        self.labels[self.current_frame] = label
        self.update_display()
    
    def on_slider_change(self, val):
        """Handle slider change."""
        self.current_frame = int(val)
        self.update_display()
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.key == 'left':
            self.navigate(-1)
        elif event.key == 'right':
            self.navigate(1)
        elif event.key == 'up':
            self.navigate(-10)
        elif event.key == 'down':
            self.navigate(10)
        elif event.key == '0':
            self.set_label(0)  # Failure
        elif event.key == '1':
            self.set_label(1)  # Success
        elif event.key == '-':
            self.set_label(-1)  # Ignore
    
    def on_timeline_click(self, event):
        """Handle timeline click to jump to frame."""
        if event.inaxes == self.ax_timeline:
            frame_idx = int(event.xdata)
            if 0 <= frame_idx < self.num_frames:
                self.current_frame = frame_idx
                self.slider.set_val(self.current_frame)
                self.update_display()
    
    # Bulk labeling button functions
    def mark_success_start(self, event):
        """Mark current frame as success start."""
        self.success_start = self.current_frame
        print(f"SUCCESS START marked at frame {self.current_frame}")
        
    def mark_success_end(self, event):
        """Mark current frame as success end and label the range."""
        if self.success_start is not None:
            self.success_end = self.current_frame
            start = min(self.success_start, self.success_end)
            end = max(self.success_start, self.success_end)
            self.labels[start:end+1] = 1
            self.update_display()
            print(f"SUCCESS: Labeled frames {start}-{end} as SUCCESS")
        else:
            print("Please mark SUCCESS START first!")
    
    def add_trailing_success(self, event):
        """Add trailing success frames after success end."""
        if self.success_end is not None:
            trailing = 50  # Default trailing frames
            if self.success_end + trailing < self.num_frames:
                self.labels[self.success_end+1:self.success_end+1+trailing] = 1
                print(f"Added {trailing} trailing success frames")
            else:
                remaining = self.num_frames - self.success_end - 1
                self.labels[self.success_end+1:] = 1
                print(f"Added {remaining} trailing success frames (to end)")
            self.update_display()
        else:
            print("Please mark SUCCESS END first!")
    
    def mark_ignore_start(self, event):
        """Mark current frame as ignore start."""
        self.ignore_start = self.current_frame
        print(f"IGNORE START marked at frame {self.current_frame}")
        
    def mark_ignore_end(self, event):
        """Mark current frame as ignore end and label the range."""
        if self.ignore_start is not None:
            self.ignore_end = self.current_frame
            start = min(self.ignore_start, self.ignore_end)
            end = max(self.ignore_start, self.ignore_end)
            self.labels[start:end+1] = -1
            self.update_display()
            print(f"IGNORE: Labeled frames {start}-{end} as IGNORE")
        else:
            print("Please mark IGNORE START first!")
    
    def mark_failure_start(self, event):
        """Mark current frame as failure start."""
        self.failure_start = self.current_frame
        print(f"FAILURE START marked at frame {self.current_frame}")
        
    def mark_failure_end(self, event):
        """Mark current frame as failure end and label the range."""
        if self.failure_start is not None:
            self.failure_end = self.current_frame
            start = min(self.failure_start, self.failure_end)
            end = max(self.failure_start, self.failure_end)
            self.labels[start:end+1] = 0
            self.update_display()
            print(f"FAILURE: Labeled frames {start}-{end} as FAILURE")
        else:
            print("Please mark FAILURE START first!")
    
    def clear_all_labels(self, event):
        """Clear all labels."""
        self.labels.fill(0)  # Reset to all failure
        self.update_display()
        print("Cleared all labels (reset to FAILURE)")
    
    def save_labels(self, event):
        """Save current labels."""
        self.saved = True
        success_count = np.sum(self.labels == 1)
        failure_count = np.sum(self.labels == 0)
        ignore_count = np.sum(self.labels == -1)
        total = len(self.labels)
        
        print(f"\n💾 SAVED Episode {self.episode_idx} Labels:")
        print(f"  Total frames: {total}")
        print(f"  Success: {success_count} ({success_count/total*100:.1f}%)")
        print(f"  Failure: {failure_count} ({failure_count/total*100:.1f}%)")
        print(f"  Ignore: {ignore_count} ({ignore_count/total*100:.1f}%)")
        print("  Labels saved! You can now close the window or go to next episode.")
        print("")
    
    def next_episode(self, event):
        """Request to go to next episode."""
        if not self.saved:
            print("⚠️  WARNING: You haven't saved your labels yet!")
            print("   Click 'Save' first, then 'Next Episode'")
            return
        
        self.next_episode_requested = True
        print(f"✅ Moving to next episode...")
        plt.close(self.fig)  # Close current window
    def update_display(self):
        """Update all displays."""
        self.update_image()
        self.update_timeline()
        self.fig.canvas.draw()
    
    def get_labels(self):
        """Get the final labels."""
        return self.labels.copy()
    
    def get_result(self):
        """Get labeling results and next episode request."""
        return {
            'labels': self.labels.copy(),
            'saved': self.saved,
            'next_episode': self.next_episode_requested
        }
    
    def show(self):
        """Show the labeling interface."""
        print("\n" + "="*60)
        print("FRAME LABELING INSTRUCTIONS:")
        print("")
        print("NAVIGATION:")
        print("  - Left/Right arrows: Navigate 1 frame")
        print("  - Up/Down arrows: Jump 10 frames") 
        print("  - Slider: Jump to any frame")
        print("  - Click timeline: Jump to specific frame")
        print("")
        print("SINGLE FRAME LABELING (Bottom Row):")
        print("  - Press 0 or 'Failure' button: Mark current frame as FAILURE (red)")
        print("  - Press 1 or 'Success' button: Mark current frame as SUCCESS (green)")
        print("  - Press - or 'Ignore' button: Mark current frame as IGNORE (gray)")
        print("")
        print("BULK LABELING (Top Row):")
        print("  SUCCESS RANGE:")
        print("    1. Navigate to start frame → click 'Mark Success Start'")
        print("    2. Navigate to end frame → click 'Mark Success End'")
        print("    3. Click 'Add Trailing 50' to add trailing success frames")
        print("")
        print("  IGNORE RANGE:")
        print("    1. Navigate to start frame → click 'Mark Ignore Start'")
        print("    2. Navigate to end frame → click 'Mark Ignore End'")
        print("")
        print("  FAILURE RANGE:")
        print("    1. Navigate to start frame → click 'Mark Fail Start'")
        print("    2. Navigate to end frame → click 'Mark Fail End'")
        print("")
        print("UTILITIES:")
        print("  - 'Clear All': Reset all labels to failure")
        print("  - 'Save': Save current labels (required before next episode)")
        print("  - 'Next Episode': Go to next episode (must save first)")
        print("")
        print("Close window when done labeling (or use Next Episode button)")
        print("="*60 + "\n")
        
        plt.show()
        return self.get_result()


def label_episode_frames(dataset_repo_id, episode_idx):
    """Label frames for a specific episode."""
    print(f"Loading dataset: {dataset_repo_id}")
    dataset = LeRobotDataset(dataset_repo_id)
    
    if episode_idx >= dataset.num_episodes:
        raise ValueError(f"Episode {episode_idx} not found. Dataset has {dataset.num_episodes} episodes.")
    
    print(f"Starting frame-level labeling for episode {episode_idx}")
    labeler = FrameLevelLabeler(dataset, episode_idx)
    result = labeler.show()
    
    return result


def create_frame_labeled_dataset_fast(original_repo_id, episode_labels_dict, output_repo_id):
    """Create a new dataset with frame-level labels - optimized version."""
    print(f"Creating frame-labeled dataset: {original_repo_id} -> {output_repo_id}")
    
    # Create output directory
    output_dir = Path("data") / output_repo_id.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating frame-labeled dataset in: {output_dir}")
    
    # Load original dataset metadata only (no video decoding)
    original_dataset = LeRobotDataset(original_repo_id)
    
    # Create labeled dataset using LeRobotDataset.create
    labeled_dataset = LeRobotDataset.create(
        output_repo_id,
        original_dataset.fps,
        root="data",
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        features=original_dataset.features,  # Use same features as original
    )
    
    # Process each episode that was labeled
    total_frames = 0
    for episode_idx, labels in episode_labels_dict.items():
        print(f"Processing episode {episode_idx} with {len(labels)} frames...")
        
        # Get episode frame indices
        from_idx = original_dataset.episode_data_index["from"][episode_idx].item()
        to_idx = original_dataset.episode_data_index["to"][episode_idx].item()
        
        # Process each frame in the episode
        for frame_idx in range(from_idx, to_idx):
            # Get original frame data
            original_frame = original_dataset[frame_idx]
            
            # Calculate position within episode
            episode_frame_idx = frame_idx - from_idx
            
            if episode_frame_idx >= len(labels):
                print(f"Warning: Frame {frame_idx} (episode {episode_idx}, pos {episode_frame_idx}) beyond labels length {len(labels)}")
                continue
            
            # Get label for this frame
            frame_label = labels[episode_frame_idx]
            
            # Create labeled frame data
            labeled_frame = {}
            for key, value in original_frame.items():
                if key != "next.reward":  # Don't copy original reward
                    labeled_frame[key] = value
            
            # Add frame-level reward label
            labeled_frame["next.reward"] = torch.tensor([float(frame_label)], dtype=torch.float32)
            
            # Add to dataset
            labeled_dataset.add_frame(labeled_frame)
            total_frames += 1
    
    # Finalize dataset
    labeled_dataset.finalize()
    
    print(f"✅ Created frame-labeled dataset with {total_frames} frames")
    print(f"📁 Saved to: {output_dir}")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Frame-level labeling for reward classifier training")
    parser.add_argument("--dataset_repo_id", type=str, required=True, 
                       help="HuggingFace repo ID of the teleop dataset")
    parser.add_argument("--episode_idx", type=int, default=0,
                       help="Episode index to label (default: 0)")
    parser.add_argument("--all_episodes", action="store_true",
                       help="Label all episodes interactively")
    parser.add_argument("--output_repo_id", type=str, 
                       help="Output repo ID for labeled dataset")
    
    args = parser.parse_args()
    
    dataset = LeRobotDataset(args.dataset_repo_id)
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"Episodes: {dataset.num_episodes}")
    
    episode_labels_dict = {}
    
    if args.all_episodes:
        # Label all episodes
        current_ep = 0
        while current_ep < dataset.num_episodes:
            print(f"\n--- Labeling Episode {current_ep + 1}/{dataset.num_episodes} ---")
            try:
                result = label_episode_frames(args.dataset_repo_id, current_ep)
                
                if result['saved']:
                    episode_labels_dict[current_ep] = result['labels']
                    
                    # Show statistics
                    labels = result['labels']
                    success_count = np.sum(labels == 1)
                    failure_count = np.sum(labels == 0)
                    ignore_count = np.sum(labels == -1)
                    print(f"✅ Episode {current_ep} completed: {success_count} success, {failure_count} failure, {ignore_count} ignore frames")
                    
                    if result['next_episode']:
                        current_ep += 1  # Go to next episode
                    else:
                        print("Labeling session ended by user.")
                        break
                else:
                    print(f"❌ Episode {current_ep} not saved. Skipping...")
                    current_ep += 1
                    
            except Exception as e:
                print(f"Error labeling episode {current_ep}: {e}")
                current_ep += 1
                continue
    else:
        # Label single episode (or start from specific episode and continue)
        current_ep = args.episode_idx
        while current_ep < dataset.num_episodes:
            print(f"\n--- Labeling Episode {current_ep + 1}/{dataset.num_episodes} ---")
            try:
                result = label_episode_frames(args.dataset_repo_id, current_ep)
                
                if result['saved']:
                    episode_labels_dict[current_ep] = result['labels']
                    
                    # Show statistics
                    labels = result['labels']
                    success_count = np.sum(labels == 1)
                    failure_count = np.sum(labels == 0)
                    ignore_count = np.sum(labels == -1)
                    print(f"✅ Episode {current_ep} completed: {success_count} success, {failure_count} failure, {ignore_count} ignore frames")
                    
                    if result['next_episode'] and current_ep + 1 < dataset.num_episodes:
                        current_ep += 1  # Go to next episode
                    else:
                        if current_ep + 1 >= dataset.num_episodes:
                            print("🎉 All episodes completed!")
                        else:
                            print("Labeling session ended by user.")
                        break
                else:
                    print(f"❌ Episode {current_ep} not saved.")
                    break
                    
            except Exception as e:
                print(f"Error labeling episode {current_ep}: {e}")
                break
    
    # Create labeled dataset
    if episode_labels_dict and args.output_repo_id:
        print(f"\n🎯 Creating frame-labeled dataset from {len(episode_labels_dict)} episodes...")
        output_dir = create_frame_labeled_dataset_fast(
            args.dataset_repo_id, 
            episode_labels_dict, 
            args.output_repo_id
        )
        print(f"✅ Frame-labeled dataset created: {output_dir}")
        
        # Print summary statistics
        print(f"\n📊 LABELING SUMMARY:")
        total_success = 0
        total_failure = 0
        total_ignore = 0
        total_frames = 0
        
        for ep_idx, labels in episode_labels_dict.items():
            success_count = np.sum(labels == 1)
            failure_count = np.sum(labels == 0)
            ignore_count = np.sum(labels == -1)
            total_success += success_count
            total_failure += failure_count
            total_ignore += ignore_count
            total_frames += len(labels)
            
            print(f"  Episode {ep_idx}: {success_count} success, {failure_count} failure, {ignore_count} ignore")
        
        print(f"\n📈 TOTAL: {total_success} success, {total_failure} failure, {total_ignore} ignore")
        print(f"   Success rate: {total_success/total_frames*100:.1f}%")
        print(f"   Dataset ready for reward classifier training! 🚀")
        
    else:
        print("❌ No labeled episodes or no output repo specified")


if __name__ == "__main__":
    main()