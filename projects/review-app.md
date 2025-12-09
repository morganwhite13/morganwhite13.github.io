---
layout: page
title: Morgan's Reviews - iOS Entertainment Review App
permalink: /projects/review-app/
---

## Quick Summary

A native iOS review application built in Swift that allows a reviewer to create, edit, and publish reviews of movies, TV shows, games, and other entertainment. Users can browse reviews with ratings, images, and detailed commentary, while the reviewer maintains editorial control through a secure sign-in system.

**Tech Stack:** Swift, UIKit, Xcode, iOS SDK  
**Status:** ✅ Complete and Deployed (2019)  
**Platform:** iOS (iPhone/iPad)

---

## Video Demo

<iframe width="100%" height="450" src="[https://youtu.be/](https://www.youtube.com/embed/sYmV_4kQiN8)" title="Review App Project Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

---

## The Challenge

Design and implement a dual-interface iOS application that serves two distinct user groups:
- **Reviewers** (authenticated): Create, edit, and delete reviews with full editorial control
- **General Users**: Browse and read published reviews in a clean, intuitive interface

The app needed to provide:
- Secure authentication preventing unauthorized edits
- Rich media support (images from photo library)
- Custom star rating system with accessibility features
- Persistent storage of review content
- Seamless navigation between viewing and editing modes

---

## System Architecture

### Application Structure

The app follows the Model-View-Controller (MVC) pattern with seven main components:

**Models:**
- `Review.swift` - Core data model with validation logic

**Views:**
- `RatingControl.swift` - Custom UIStackView-based star rating component
- `ReviewTableViewCell.swift` - Custom table cell for review list

**Controllers:**
- `ReviewTableViewController.swift` - Main list view (7 pre-loaded reviews)
- `ReviewUserrViewController.swift` - Read-only review detail view
- `ReviewViewController.swift` - Full edit/create review interface
- `SigninViewController.swift` - Reviewer authentication for new reviews
- `SigninUserViewController.swift` - Reviewer authentication for editing existing reviews

**Navigation Flow:**
```
ReviewTableViewController (Main List)
    ├── [Tap Review] → ReviewUserrViewController (Read-Only)
    │                   └── [Edit Button] → SigninUserViewController
    │                                       └── [Auth Success] → ReviewViewController
    └── [+ Button] → SigninViewController
                     └── [Auth Success] → ReviewViewController
```

---

## Key Features

### 1. Custom Star Rating System

Built a reusable `RatingControl` component using UIStackView:

**Features:**
- 5-star rating scale (0-5)
- Three button states: empty, filled, highlighted
- Toggle functionality (tap same rating to reset to 0)
- VoiceOver accessibility support
- Configurable star count and size via `@IBInspectable`
- User interaction enable/disable for read-only mode

**Implementation Highlights:**
- Dynamic button array management
- Custom image asset loading from bundle
- Programmatic constraint setup
- State synchronization with `didSet` property observer

### 2. Dual Authentication System

Implemented two separate sign-in flows to handle different use cases:

**SigninViewController** (New Reviews):
- Simple credential validation (username: "good", password: "app")
- Enables "Done" button only after successful authentication
- Direct segue to `ReviewViewController` for creation

**SigninUserViewController** (Edit Existing):
- Receives review object from `ReviewUserrViewController`
- Passes review data to `ReviewViewController` upon authentication
- Maintains review context through navigation chain

### 3. Review Management System

**Create Operation:**
- Text field validation (title and paragraph required)
- Image picker integration for photo library access
- Real-time save button state management
- Navigation title updates as title is typed

**Update Operation:**
- Pre-populates all fields with existing review data
- Enables delete button (disabled during creation)
- Preserves rating and image during edits

**Delete Operation:**
- Sets rating to 31 (sentinel value) to signal deletion
- `ReviewTableViewController` detects sentinel and removes row
- Smooth fade animation on deletion

### 4. Image Handling

Integrated UIImagePickerController for media management:

```swift
@IBAction func selectImageFromPhotoLibrary(_ sender: UITapGestureRecognizer) {
    nameTextField.resignFirstResponder()
    let imagePickerController = UIImagePickerController()
    imagePickerController.sourceType = .savedPhotosAlbum
    imagePickerController.delegate = self
    present(imagePickerController, animated: true, completion: nil)
}
```

**Features:**
- Tap gesture recognizer on UIImageView
- Dismisses keyboard before showing picker
- Extracts original unedited image
- Persistent storage in Review model

---

## Technical Implementation

### Review Data Model

```swift
class Review {
    var title: String
    var photo: UIImage?
    var rating: Int
    var reviewwords: String
    
    init?(title: String, photo: UIImage?, rating: Int, reviewwords: String) {
        guard !title.isEmpty && !reviewwords.isEmpty else {
            return nil
        }
        guard (((rating >= 0) && (rating <= 5)) || rating == 31) else {
            return nil
        }
        // Initialization...
    }
}
```

**Validation Rules:**
- Title and review text cannot be empty
- Rating must be 0-5 (or 31 for deletion sentinel)
- Failable initializer returns `nil` for invalid data

### Table View Data Source

The main table view controller manages a dynamic array of reviews:

**Cell Configuration:**
```swift
override func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
    guard let cell = tableView.dequeueReusableCell(withIdentifier: cellIdentifier, for: indexPath) as? ReviewTableViewCell else {
        fatalError("The dequeued cell is not an instance of ReviewTableViewCell")
    }
    let review = reviews[indexPath.row]
    cell.nameLabel.text = review.title
    cell.photoImageView.image = review.photo
    cell.ratingControl.rating = review.rating
    cell.backgroundColor = UIColor(red: 0/255.0, green: 255/255.0, blue: 127/255.0, alpha: 1.0)
    return cell
}
```

### Unwind Segue Logic

Implemented sophisticated unwind logic to handle all navigation scenarios:

```swift
@IBAction func unwindToReviewList(sender: UIStoryboardSegue) {
    if let sourceViewController = sender.source as? ReviewViewController, let review = sourceViewController.review {
        if let selectedIndexPath = tableView.indexPathForSelectedRow {
            if review.rating == 31 {
                // Delete operation
                reviews.remove(at: selectedIndexPath.row)
                tableView.deleteRows(at: [selectedIndexPath], with: .fade)
            } else {
                // Update operation
                reviews[selectedIndexPath.row] = review
                tableView.reloadRows(at: [selectedIndexPath], with: .none)
            }
        } else {
            // Create operation
            let newIndexPath = IndexPath(row: reviews.count, section: 0)
            reviews.append(review)
            tableView.insertRows(at: [newIndexPath], with: .automatic)
        }
    }
}
```

**Smart Detection:**
- Checks if `selectedIndexPath` exists to determine create vs. edit
- Uses rating sentinel (31) to detect delete operations
- Applies appropriate animation for each operation type

---

## User Interface Design

### Visual Identity

**Color Scheme:**
- Primary: Spring Green (RGB: 0, 255, 127)
- Applied consistently across all view controllers
- High contrast for readability

**Typography & Layout:**
- Standard iOS fonts for familiarity
- Clear visual hierarchy (title → image → rating → body text)
- Ample spacing for comfortable reading

### Accessibility Features

**VoiceOver Support in RatingControl:**
```swift
button.accessibilityLabel = "Set \(index + 1) star rating"
button.accessibilityValue = "\(rating) stars set"
button.accessibilityHint = rating == index + 1 ? "Tap to reset the rating to zero" : nil
```

**Features:**
- Descriptive labels for each star button
- Current rating announcement
- Reset hint when tapping active rating
- Screen reader friendly navigation

---

## Pre-Loaded Content

The app ships with 7 sample reviews covering various entertainment:

| Title | Rating | Category |
|-------|--------|----------|
| DC's best movie yet: Shazam! | ⭐⭐⭐⭐⭐ | Movie |
| The Biggest Error in Avengers Endgame! | ⭐⭐⭐⭐ | Movie |
| John Wick 3 and the Endless Action! | ⭐⭐⭐⭐ | Movie |
| Black Ops 4 Where are the Changes? | ⭐⭐ | Game |
| Morgan's Reviews the Best App Ever | ⭐⭐⭐⭐⭐ | App |
| Computer Science the best subject | ⭐⭐⭐⭐⭐ | Education |
| SuperStore the soon to be superstar show | ⭐⭐⭐⭐ | TV Show |

---

## Challenges & Solutions

### Challenge: Dual User Interface Paradigms

**Problem:** The app needed to serve both authenticated reviewers with full CRUD capabilities and general users with read-only access, without confusing the two experiences.

**Solution:** 
- Created separate view controllers for read-only (`ReviewUserrViewController`) and editable (`ReviewViewController`) interfaces
- Disabled star rating interaction in user view: `userrating.isUserInteractionEnabled = false`
- Implemented two authentication flows to handle creating new reviews vs. editing existing ones
- Used segue chains to maintain context through authentication

**Result:** Clean separation of concerns with intuitive navigation for both user types

---

### Challenge: Review State Persistence Through Navigation

**Problem:** Reviews needed to maintain their state when passed through multiple view controllers (TableView → UserView → SignIn → EditView)

**Solution:**
```swift
// In ReviewUserrViewController
override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
    guard let reviewDetailViewController = segue.destination as? SigninUserViewController else {
        fatalError("Unexpected Destination: \(segue.destination)")
    }
    let selectedReview = self.review
    reviewDetailViewController.review = selectedReview
}
```

**Pattern Used:**
- Pass review object through each segue's `prepare(for:sender:)`
- Maintain optional `var review: Review?` in each controller
- Type-safe unwrapping with guard statements

**Result:** Seamless data flow through complex navigation chains

---

### Challenge: Delete Operation Without Explicit Delete Button in Table

**Problem:** Standard iOS pattern uses swipe-to-delete, but app needed delete functionality only for authenticated reviewers within the edit interface.

**Solution:**
- Added delete button in `ReviewViewController` (only visible when editing)
- Used sentinel value (rating = 31) to signal deletion intent
- `unwindToReviewList` checks rating and removes row if sentinel detected
- Preserved table view's clean interface without exposed delete actions

**Code Pattern:**
```swift
// In ReviewViewController
if let button = sender as? UIButton, button === delete {
    review = Review(title: title, photo: photo, rating: 31, reviewwords: reviewwords)
}

// In ReviewTableViewController
if review.rating == 31 {
    reviews.remove(at: selectedIndexPath.row)
    tableView.deleteRows(at: [selectedIndexPath], with: .fade)
}
```

**Result:** Secure delete functionality without compromising user interface simplicity

---

### Challenge: Save Button State Management

**Problem:** Users could potentially save incomplete reviews without both title and review text.

**Solution:**
- Implemented `updateSaveButtonState()` method
- Text field delegate monitors editing state
- Disables save button during active editing
- Re-enables only when both fields contain text

```swift
private func updateSaveButtonState() {
    let text = nameTextField.text ?? ""
    let text2 = paragraph.text ?? ""
    saveButton.isEnabled = !text.isEmpty && !text2.isEmpty
}

func textFieldDidBeginEditing(_ textField: UITextField) {
    saveButton.isEnabled = false
}

func textFieldDidEndEditing(_ textField: UITextField) {
    updateSaveButtonState()
}
```

**Result:** Prevents invalid review submissions while providing clear visual feedback

---

## Code Architecture

### File Structure (600+ lines total)

```
morgansreview/
├── Models/
│   └── Review.swift (30 lines)
├── Views/
│   ├── RatingControl.swift (130 lines)
│   └── ReviewTableViewCell.swift (25 lines)
├── Controllers/
│   ├── ReviewTableViewController.swift (140 lines)
│   ├── ReviewUserrViewController.swift (50 lines)
│   ├── ReviewViewController.swift (140 lines)
│   ├── SigninViewController.swift (45 lines)
│   └── SigninUserViewController.swift (50 lines)
└── AppDelegate.swift (40 lines)
```

### Design Patterns Used

**Model-View-Controller (MVC):**
- Clean separation between data (Review), presentation (Custom views), and logic (Controllers)

**Delegation:**
- UITextFieldDelegate for keyboard management
- UIImagePickerControllerDelegate for photo selection
- UITableViewDelegate/DataSource for list management

**Target-Action:**
- Button taps in RatingControl
- Bar button items in navigation

**Optional Chaining:**
- Safe unwrapping throughout: `let text = nameTextField.text ?? ""`
- Guard statements for type-safe downcasting

**Failable Initializers:**
- Review model validates data at creation time
- Returns nil for invalid input

---

## What I Learned

This project taught me:

**iOS Development Fundamentals**
- UIKit framework and view hierarchy
- Storyboard-based UI design with programmatic constraints
- Navigation controller stack management
- Delegate pattern and protocol conformance

**Swift Language Features**
- Optional handling and guard statements
- Property observers (didSet, willSet)
- Failable initializers for validation
- Type safety and casting

**User Experience Design**
- Authentication flow design
- State management across view controllers
- Accessibility considerations (VoiceOver)
- Visual feedback for user actions

**Table View Mastery**
- Custom cell creation and registration
- Dynamic row insertion/deletion with animation
- Cell reuse and performance optimization
- Section and row management

**Image Handling**
- UIImagePickerController integration
- Photo library permissions
- Image asset management in bundle
- Button state images

---

## Original Project Goals vs. Reality

### From the Proposal (April 2019):

**Original Vision:**
- Review app for movies, TV shows, games, products, books
- Alternative to trailers that give away plots
- Help people find suitable entertainment
- Build reviews based on personality matching

**What Was Achieved:**
- ✅ Multi-category review system (movies, shows, games)
- ✅ Honest, spoiler-free review format
- ✅ Clean, article-style presentation
- ✅ Star rating system for quick assessment
- ✅ Rich media support with images
- ✅ Editorial control through authentication

**What Was Simplified:**
- ❌ Public user-submitted reviews (remained single-reviewer focused)
- ❌ Personality-based recommendation algorithm
- ❌ Cross-platform deployment (iOS only)

**Reflection:**
The simplification from the original proposal was intentional and wise for a school project timeline. The focus shifted from a social review platform to a personal review blog in app format—a more achievable scope that still demonstrated all core iOS development skills.

---

## Future Improvements

If I were to extend this project today, I would:

1. **Firebase Backend Integration** - Replace in-memory storage with cloud database for true persistence
2. **SwiftUI Rewrite** - Modernize using declarative UI framework
3. **User Accounts** - Multiple reviewers with profiles
4. **Search & Filtering** - Find reviews by category, rating, or keyword
5. **Rich Text Editor** - Markdown support for formatted reviews
6. **Share Functionality** - Export reviews to social media
7. **Dark Mode Support** - Respect system appearance preferences
8. **iPad Optimization** - Split-view layout for larger screens
9. **Localization** - Multi-language support
10. **Analytics** - Track most-viewed reviews and user engagement

---

## Technologies Used

**Languages & Frameworks:**
- Swift 5.0
- UIKit
- Foundation

**Development Tools:**
- Xcode 10.2
- Interface Builder (Storyboards)
- iOS Simulator

**Key APIs:**
- UITableView & UITableViewController
- UIImagePickerController
- UINavigationController
- UITextField & UITextFieldDelegate
- Auto Layout (NSLayoutConstraint)
- Bundle & Asset Catalog

**Design Patterns:**
- Model-View-Controller (MVC)
- Delegation
- Target-Action
- Observer (Property Observers)

---

## Project Timeline

**April 2019:**
- Week 1-2: Proposal, design, and Swift learning
- Week 3: Core Review model and RatingControl implementation
- Week 4: Table view and navigation setup

**May 2019:**
- Week 1: Authentication system and edit/create flows
- Week 2-3: Polish, testing, and pre-loaded content
- Week 4: App Fair presentation and final submission

**Total Development Time:** ~6 weeks (part-time, during school)

---

## Takeaway

This project demonstrates end-to-end iOS application development: from initial proposal and UX planning, through implementation of custom UI components and complex navigation flows, to deployment of a polished, functional app. It showcases my ability to work with iOS frameworks, implement secure authentication patterns, manage complex state across multiple view controllers, and deliver a user-friendly solution within a school project timeline.

The app successfully achieved its core goal: providing a platform for honest, spoiler-free entertainment reviews that help users make informed viewing decisions—all wrapped in a clean, accessible iOS interface.
