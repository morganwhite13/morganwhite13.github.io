---
layout: page
title: Morgan's Reviews - iOS Entertainment Tracker
permalink: /projects/morgans-reviews/
---

# Morgan's Reviews iOS App

## Quick Summary

A native iOS application designed to track and review entertainment media (movies, games, TV shows) without spoilers. Built using Swift and UIKit, the app features a custom-built interactive rating system, a secure admin interface for content management, and dynamic table views for organizing reviews.

**Tech Stack:** Swift 5, UIKit, Xcode, Interface Builder, MVC Pattern  
**Status:** ✅ Prototype Complete (Spring 2019)  
**Repository:** [View Source Code](https://github.com/yourusername/morgans-reviews)

---

## The Challenge

Finding reliable entertainment recommendations is often difficult. Trailers frequently contain spoilers-Phase 1.docx], and recommendations from friends aren't always personalized. The goal was to create a dedicated platform for "honest reviews"-Phase 1.docx] where users can find spoil-free summaries and ratings for content ranging from *Avengers: Endgame* to *Call of Duty*.

## Application Architecture (MVC)

The application follows the **Model-View-Controller (MVC)** design pattern, a staple of iOS development:


* **Model:** The `Review` class defines the data structure, requiring a title, photo, rating, and review text.
* **View:** Custom `UITableViewCell` layouts and a highly customized `UIStackView` for the star rating system.
* **Controller:** `ReviewTableViewController` manages the data flow, while `ReviewViewController` handles the creation and editing logic.

---

## Key Technical Features

### 1. Custom Interactive Rating Control
Instead of using a standard slider or picker, I built a custom UI component from scratch. The `RatingControl` class is an `@IBDesignable` subclass of `UIStackView` that programmatically generates star buttons.

* **Logic:** It manages an array of `UIButton` objects.
* **Interactivity:** The `ratingButtonTapped` function calculates the rating based on the button index. Tapping the current rating resets it to zero.
* **Accessibility:** Fully supports VoiceOver with accessibility hints (e.g., "Tap to reset the rating to zero").

```swift
// Snippet from RatingControl.swift
@objc func ratingButtonTapped(button:UIButton){
    guard let index = ratingButtons.firstIndex(of: button) else{
        fatalError("The button, \(button), is not in the ratingButtons array")
    }
    let selectedRating = index + 1
    if selectedRating == rating {
        rating = 0
    } else {
        rating = selectedRating
    }
}
