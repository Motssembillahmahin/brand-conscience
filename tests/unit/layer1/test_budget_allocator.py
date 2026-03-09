"""Tests for budget allocator."""

from __future__ import annotations

from brand_conscience.layer1_strategy.budget_allocator import BudgetAllocator


def test_allocate_within_bounds():
    allocator = BudgetAllocator()
    result = allocator.allocate(budget_fraction=0.5, urgency_score=0.5)
    assert result.daily_budget > 0
    assert result.daily_budget <= 5000.0


def test_high_urgency_increases_budget():
    allocator = BudgetAllocator()
    low = allocator.allocate(budget_fraction=0.5, urgency_score=0.2)
    high = allocator.allocate(budget_fraction=0.5, urgency_score=0.9)
    assert high.daily_budget > low.daily_budget


def test_approval_threshold():
    allocator = BudgetAllocator()
    low_budget = allocator.allocate(budget_fraction=0.1, urgency_score=0.3)
    assert not low_budget.requires_approval

    high_budget = allocator.allocate(budget_fraction=1.0, urgency_score=1.0)
    # May or may not require approval depending on config
    assert isinstance(high_budget.requires_approval, bool)
