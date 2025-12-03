---
layout: post
title: "Introducing Morgan's Reviews: A Spoiler-Free Zone"
date: 2025-11-28 09:00:00 -0500
categories: ios swift projects
---

Have you ever watched a movie trailer that gave away the entire plot? Or gotten a recommendation from a friend that just didn't match your taste?

I built my latest iOS project, **Morgan's Reviews**, to solve exactly this problem. It is a dedicated platform for finding honest entertainment reviews without the risk of spoilers ruining the experience-Phase 1.docx].

## The Problem with Modern Trailers

When I started designing this app, I realized that "more often than not, trailers give away the majority of the plots of movies"-Phase 1.docx]. This defeats the purpose of watching the movie in the first place.

My goal was to create an app that provides "honest reviews on various types of entertainment," from movies like *Shazam!* and *Avengers: Endgame* to games like *Black Ops 4*-Phase 1.docx, ReviewTableViewController.swift].

## Inside the App

**Morgan's Reviews** is a native iOS application built using Swift and UIKit. It follows the Model-View-Controller (MVC) pattern to keep the code organized and scalable.

Here are a few of the key features I implemented:

### 1. A Custom Rating System
I wanted the rating experience to be intuitive. I built a custom `RatingControl` that allows users to tap on stars to set a score from 0 to 5. It even supports accessibility features, providing voice hints like "Tap to reset the rating to zero" for visually impaired users.

### 2. Admin vs. User Modes
To ensure the integrity of the reviews, I separated the application into two distinct modes:
* **Reader Mode:** The default view where users can browse and read reviews but cannot alter the ratings.
* **Creator Mode:** A protected area where I can log in with specific credentials to add new content, upload photos from the library, or delete reviews.

### 3. Dynamic Content
The app handles a variety of media types. Whether it's a TV show like *Superstore* or a video game, the app dynamically adjusts the table view to display the relevant cover art and rating summaries.

## What I Learned

Building this prototype taught me a lot about iOS navigation and data passing. One of the most interesting challenges was handling the "Unwind Segue"—the logic that determines whether a review should be added as a new row, updated in place, or deleted entirely based on the user's action.

If you are looking for a trustworthy source for entertainment that respects your desire to remain spoiler-free, this is the app for you.

[Check out the full project details here](/projects/morgans-reviews/)
